import pandas as pd
import streamlit as st
import time
import logging
from collections import defaultdict
from rapidfuzz import process, fuzz
from datetime import datetime

logger = logging.getLogger(__name__)


class AdvancedAddressMatcher:
    def __init__(self, spr_df, cad_df, max_records=None):
        # Limit SPR records if specified
        if max_records is not None and max_records < len(spr_df):
            self.spr_df = spr_df.head(max_records).copy()
            st.info(f"📊 Processing limited to {max_records:,} SPR records out of {len(spr_df):,} total records")
        else:
            self.spr_df = spr_df.copy()

        self.cad_df = cad_df
        self.original_spr_count = len(spr_df)
        self.processing_count = len(self.spr_df)

        # Create indices based on the limited dataset
        self.street_index = self._create_street_index()
        self.house_index = self._create_house_index()
        self.search_key_index = self._create_search_key_index()

    def _create_street_index(self):
        """Create street-based index for fast lookups"""
        index = defaultdict(list)
        for idx, row in self.cad_df.iterrows():
            street = row['STREET_NORM']
            if street:
                index[street].append(idx)
        return dict(index)

    def _create_house_index(self):
        """Create house-based index for fast lookups"""
        index = defaultdict(list)
        for idx, row in self.cad_df.iterrows():
            house = row['HOUSE_NORM']
            if house:
                index[house].append(idx)
        return dict(index)

    def _create_search_key_index(self):
        """Create composite search key index"""
        index = defaultdict(list)
        for idx, row in self.cad_df.iterrows():
            key = row['SEARCH_KEY']
            if key:
                index[key].append(idx)
        return dict(index)

    def find_exact_matches(self):
        """Find exact matches using multiple strategies with detailed progress"""
        matches = []
        total_spr = len(self.spr_df)

        # Create progress tracking components
        progress_container = st.container()
        with progress_container:
            st.subheader("🎯 Exact Matching Progress")
            progress_bar = st.progress(0)
            status_text = st.empty()
            stats_cols = st.columns(4)

            with stats_cols[0]:
                processed_metric = st.empty()
            with stats_cols[1]:
                matches_metric = st.empty()
            with stats_cols[2]:
                rate_metric = st.empty()
            with stats_cols[3]:
                eta_metric = st.empty()

        start_time = time.time()

        # Strategy 1: Full address exact match
        status_text.text("🔍 Phase 1: Full address exact matching...")
        cad_full_lookup = {row['FULL_ADDRESS']: idx for idx, row in self.cad_df.iterrows()}

        chunk_size = 1000
        phase1_matches = 0

        for i in range(0, total_spr, chunk_size):
            if st.session_state.get('stop_requested', False):
                break

            chunk = self.spr_df.iloc[i:i + chunk_size]
            chunk_matches = 0

            for idx, spr_row in chunk.iterrows():
                full_address = spr_row['FULL_ADDRESS']
                if full_address in cad_full_lookup:
                    cad_idx = cad_full_lookup[full_address]
                    matches.append(self._create_match_record(spr_row, self.cad_df.iloc[cad_idx], 100, "EXACT_FULL"))
                    chunk_matches += 1
                    phase1_matches += 1

            # Update progress
            processed = min(i + chunk_size, total_spr)
            progress = processed / total_spr
            elapsed_time = time.time() - start_time

            # Calculate ETA
            if processed > 0:
                rate = processed / elapsed_time
                eta = (total_spr - processed) / rate if rate > 0 else 0
            else:
                eta = 0

            # Update UI
            progress_bar.progress(progress)
            status_text.text(
                f"Phase 1: Processed {processed:,}/{total_spr:,} records | Found {chunk_matches} matches in current chunk")

            processed_metric.metric("Processed", f"{processed:,}/{total_spr:,}")
            matches_metric.metric("Phase 1 Matches", f"{phase1_matches:,}")
            rate_metric.metric("Rate", f"{rate:.1f} records/sec" if rate > 0 else "Calculating...")
            eta_metric.metric("ETA", f"{eta:.1f}s" if eta > 0 else "Calculating...")

        # Strategy 2: Search key exact match
        status_text.text("🔍 Phase 2: Search key exact matching...")
        phase2_matches = 0

        for i in range(0, total_spr, chunk_size):
            if st.session_state.get('stop_requested', False):
                break

            chunk = self.spr_df.iloc[i:i + chunk_size]
            chunk_matches = 0

            for idx, spr_row in chunk.iterrows():
                search_key = spr_row['SEARCH_KEY']
                if search_key in self.search_key_index:
                    for cad_idx in self.search_key_index[search_key]:
                        cad_row = self.cad_df.iloc[cad_idx]
                        if spr_row['BUILDING_NORM'] == cad_row['BUILDING_NORM']:
                            # Check if not already matched in phase 1
                            existing_match = any(
                                match['ADDRESS_ID_SPR'] == spr_row.get('ADDRESS_ID', '') and
                                match['ADDRESS_ID_CAD'] == cad_row.get('ADDRESS_ID', '')
                                for match in matches
                            )
                            if not existing_match:
                                matches.append(self._create_match_record(spr_row, cad_row, 100, "EXACT_KEY"))
                                chunk_matches += 1
                                phase2_matches += 1

            # Update progress
            processed = min(i + chunk_size, total_spr)
            progress = processed / total_spr
            elapsed_time = time.time() - start_time

            if processed > 0:
                rate = processed / elapsed_time
                eta = (total_spr - processed) / rate if rate > 0 else 0
            else:
                eta = 0

            # Update UI
            progress_bar.progress(progress)
            status_text.text(
                f"Phase 2: Processed {processed:,}/{total_spr:,} records | Found {chunk_matches} matches in current chunk")

            processed_metric.metric("Processed", f"{processed:,}/{total_spr:,}")
            matches_metric.metric("Total Matches", f"{len(matches):,} (P1: {phase1_matches}, P2: {phase2_matches})")
            rate_metric.metric("Rate", f"{rate:.1f} records/sec" if rate > 0 else "Calculating...")
            eta_metric.metric("ETA", f"{eta:.1f}s" if eta > 0 else "Complete")

        # Final update
        total_time = time.time() - start_time
        progress_bar.progress(1.0)
        status_text.text(f"✅ Exact matching completed! Found {len(matches):,} matches in {total_time:.1f}s")

        return pd.DataFrame(matches)

    def find_fuzzy_matches(self, threshold=85, chunk_size=1000, exclude_spr_ids=None):
        """Advanced fuzzy matching with multiple strategies and detailed progress"""
        if exclude_spr_ids is None:
            exclude_spr_ids = set()

        # Filter out already matched records
        remaining_spr = self.spr_df[~self.spr_df.get('ADDRESS_ID', self.spr_df.index).isin(exclude_spr_ids)]

        matches = []
        total = len(remaining_spr)

        if total == 0:
            st.info("No records remaining for fuzzy matching")
            return pd.DataFrame()

        # Create detailed progress tracking
        progress_container = st.container()
        with progress_container:
            st.subheader("🔍 Fuzzy Matching Progress")

            # Main progress bar
            main_progress = st.progress(0)

            # Status and metrics
            status_text = st.empty()

            # Metrics columns
            metrics_cols = st.columns(6)
            with metrics_cols[0]:
                processed_metric = st.empty()
            with metrics_cols[1]:
                matches_metric = st.empty()
            with metrics_cols[2]:
                rate_metric = st.empty()
            with metrics_cols[3]:
                eta_metric = st.empty()
            with metrics_cols[4]:
                candidates_metric = st.empty()
            with metrics_cols[5]:
                efficiency_metric = st.empty()

            # Detailed stats
            with st.expander("📊 Detailed Statistics", expanded=False):
                stats_cols = st.columns(3)
                with stats_cols[0]:
                    strategy_stats = st.empty()
                with stats_cols[1]:
                    score_stats = st.empty()
                with stats_cols[2]:
                    performance_stats = st.empty()

        start_time = time.time()
        total_candidates_evaluated = 0
        strategy_counts = defaultdict(int)
        score_sum = 0

        for i in range(0, total, chunk_size):
            if st.session_state.get('stop_requested', False):
                status_text.text("⏹ Stopping fuzzy matching...")
                break

            chunk = remaining_spr.iloc[i:i + chunk_size]
            chunk_matches = 0
            chunk_candidates = 0
            chunk_start_time = time.time()

            for idx, spr_row in chunk.iterrows():
                if st.session_state.get('stop_requested', False):
                    break

                # Find best match for this record
                best_match = self._find_best_fuzzy_match(spr_row, threshold)

                if best_match:
                    matches.append(best_match)
                    chunk_matches += 1
                    strategy_counts[best_match['MATCH_TYPE']] += 1
                    score_sum += best_match['MATCH_SCORE']
                    chunk_candidates += best_match.get('CANDIDATES_COUNT', 0)

                total_candidates_evaluated += chunk_candidates

            # Calculate progress and performance metrics
            processed = min(i + chunk_size, total)
            progress = processed / total
            elapsed_time = time.time() - start_time
            chunk_time = time.time() - chunk_start_time

            # Calculate rates and ETA
            if processed > 0 and elapsed_time > 0:
                overall_rate = processed / elapsed_time
                eta = (total - processed) / overall_rate if overall_rate > 0 else 0
            else:
                overall_rate = 0
                eta = 0

            chunk_rate = len(chunk) / chunk_time if chunk_time > 0 else 0
            avg_candidates_per_record = total_candidates_evaluated / processed if processed > 0 else 0
            match_efficiency = len(matches) / processed if processed > 0 else 0

            # Update main progress
            main_progress.progress(progress)

            # Update status
            status_text.text(
                f"Processing chunk {i // chunk_size + 1}/{(total - 1) // chunk_size + 1} | Found {chunk_matches} street matches in current chunk")

            # Update metrics
            processed_metric.metric("Processed", f"{processed:,}/{total:,}")
            matches_metric.metric("Street Matches", f"{len(matches):,}")
            rate_metric.metric("Rate", f"{overall_rate:.1f} rec/sec")
            eta_metric.metric("ETA", f"{eta:.0f}s" if eta > 0 else "Calculating...")
            candidates_metric.metric("Avg Candidates", f"{avg_candidates_per_record:.1f}")
            efficiency_metric.metric("Match Rate", f"{match_efficiency:.1%}")

            # Update detailed statistics
            strategy_text = "**Street Match Strategies:**\n"
            for strategy, count in strategy_counts.items():
                strategy_text += f"- {strategy}: {count}\n"
            strategy_stats.markdown(strategy_text)

            avg_score = score_sum / len(matches) if len(matches) > 0 else 0
            score_text = f"**Street Similarity Stats:**\n"
            score_text += f"- Average Score: {avg_score:.1f}\n"
            score_text += f"- Exact H+B Candidates: {total_candidates_evaluated:,}\n"
            score_text += f"- Candidates/Record: {avg_candidates_per_record:.1f}\n"
            score_stats.markdown(score_text)

            performance_text = f"**Performance:**\n"
            performance_text += f"- Chunk Rate: {chunk_rate:.1f} rec/sec\n"
            performance_text += f"- Overall Rate: {overall_rate:.1f} rec/sec\n"
            performance_text += f"- Time Elapsed: {elapsed_time:.1f}s\n"
            performance_text += f"- Est. Total Time: {elapsed_time + eta:.1f}s\n"
            performance_stats.markdown(performance_text)

            # Small delay to prevent UI from updating too rapidly
            time.sleep(0.1)

        # Final update
        total_time = time.time() - start_time
        main_progress.progress(1.0)

        final_avg_score = score_sum / len(matches) if len(matches) > 0 else 0
        final_efficiency = len(matches) / total if total > 0 else 0

        if st.session_state.get('stop_requested', False):
            status_text.text(
                f"⏹ Street fuzzy matching stopped by user. Found {len(matches):,} matches in {total_time:.1f}s")
        else:
            status_text.text(
                f"✅ Street fuzzy matching completed! Found {len(matches):,} matches in {total_time:.1f}s (avg similarity: {final_avg_score:.1f})")

        # Update final metrics
        processed_metric.metric("Processed", f"{processed:,}/{total:,}")
        matches_metric.metric("Street Matches", f"{len(matches):,}")
        rate_metric.metric("Final Rate", f"{processed / total_time:.1f} rec/sec")
        eta_metric.metric("Completed", f"{total_time:.1f}s")
        candidates_metric.metric("Avg Candidates", f"{total_candidates_evaluated / processed:.1f}")
        efficiency_metric.metric("Final Match Rate", f"{final_efficiency:.1%}")

        return pd.DataFrame(matches)

    def _find_best_fuzzy_match(self, spr_row, threshold):
        """Find best fuzzy match using street name fuzzy matching + exact house/building matching"""

        # Get exact matches first (same house and building)
        exact_house_candidates = self.house_index.get(spr_row['HOUSE_NORM'], [])

        # Filter candidates that also have exact building match
        exact_candidates = []
        for cand_idx in exact_house_candidates:
            if cand_idx < len(self.cad_df):
                cad_row = self.cad_df.iloc[cand_idx]
                if cad_row['BUILDING_NORM'] == spr_row['BUILDING_NORM']:
                    exact_candidates.append(cand_idx)

        if not exact_candidates:
            # No candidates with exact house + building match
            return None

        # Prepare street names for fuzzy matching
        candidate_streets = []
        candidate_indices = []

        for cand_idx in exact_candidates:
            cad_row = self.cad_df.iloc[cand_idx]
            street_name = cad_row['STREET_NORM']
            if street_name:  # Only include non-empty street names
                candidate_streets.append(street_name)
                candidate_indices.append(cand_idx)

        if not candidate_streets:
            return None

        # Fuzzy match only on street names
        spr_street = spr_row['STREET_NORM']
        if not spr_street:
            return None

        # Multi-strategy fuzzy matching on street names only
        strategies = [
            ('token_sort_ratio', fuzz.token_sort_ratio),
            ('token_set_ratio', fuzz.token_set_ratio),
            ('partial_ratio', fuzz.partial_ratio),
            ('ratio', fuzz.ratio)
        ]

        best_result = None
        best_score = 0
        best_strategy = None

        for strategy_name, scorer in strategies:
            result = process.extractOne(spr_street, candidate_streets, scorer=scorer)
            if result and result[1] > best_score:
                best_result = result
                best_score = result[1]
                best_strategy = strategy_name

        if best_result and best_score >= threshold:
            match_street, score, match_idx = best_result
            actual_cad_idx = candidate_indices[match_idx]
            cad_row = self.cad_df.iloc[actual_cad_idx]

            return self._create_match_record(
                spr_row, cad_row, score, f"FUZZY_STREET_{best_strategy.upper()}",
                len(exact_candidates)
            )

        return None

    def _get_candidates(self, spr_row):
        """Get candidate records for fuzzy street matching - only records with exact house+building match"""

        # Only get candidates with exact house number match
        house_candidates = set()
        house = spr_row['HOUSE_NORM']
        if house in self.house_index:
            house_candidates.update(self.house_index[house])

        # Filter candidates to only include those with exact building match
        exact_candidates = set()
        spr_building = spr_row['BUILDING_NORM']

        for cand_idx in house_candidates:
            if cand_idx < len(self.cad_df):
                cad_row = self.cad_df.iloc[cand_idx]
                if cad_row['BUILDING_NORM'] == spr_building:
                    exact_candidates.add(cand_idx)

        return exact_candidates

    def _create_match_record(self, spr_row, cad_row, score, match_type, candidates_count=1):
        """Create standardized match record"""
        return {
            "ADDRESS_ID_SPR": spr_row.get("ADDRESS_ID", ""),
            "STREET_NAME_SPR": spr_row.get("STREET_NAME", ""),
            "HOUSE_SPR": spr_row.get("HOUSE", ""),
            "BUILDING_SPR": spr_row.get("BUILDING", ""),
            "FULL_ADDRESS_SPR": spr_row.get("FULL_ADDRESS", ""),
            "ADDRESS_ID_CAD": cad_row.get("ADDRESS_ID", ""),
            "STREET_NAME_CAD": cad_row.get("STREET_NAME", ""),
            "HOUSE_CAD": cad_row.get("HOUSE", ""),
            "BUILDING_CAD": cad_row.get("BUILDING", ""),
            "FULL_ADDRESS_CAD": cad_row.get("FULL_ADDRESS", ""),
            "MATCH_SCORE": score,
            "MATCH_TYPE": match_type,
            "CANDIDATES_COUNT": candidates_count,
            "COMPLETENESS_SPR": spr_row.get("COMPLETENESS_SCORE", 0),
            "COMPLETENESS_CAD": cad_row.get("COMPLETENESS_SCORE", 0),
            "MATCH_TIMESTAMP": datetime.now().isoformat()
        }