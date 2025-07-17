import pandas as pd
from sqlalchemy import create_engine, text
from rapidfuzz import process, fuzz
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from collections import defaultdict, Counter
import re
import time
import logging
from datetime import datetime
import json
import base64
from io import BytesIO
import zipfile
from supabase import create_client, Client

# ---------------- Configuration ----------------
st.set_page_config(
    page_title="Address Registry Matcher",
    page_icon="🏘️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
    }
    .metric-container {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #2a5298;
    }
    .match-quality-excellent { background-color: #d4edda; }
    .match-quality-good { background-color: #fff3cd; }
    .match-quality-poor { background-color: #f8d7da; }
    .sidebar-section {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
    }
</style>
""", unsafe_allow_html=True)

# ---------------- Logging Setup ----------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('address_matching.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ---------------- Database Connection ----------------
@st.cache_resource
def get_supabase_client() -> Client:
    """Initialize Supabase client with error handling"""
    try:
        print(st.secrets)
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_KEY"]

        if not url or not key:
            raise ValueError("Missing SUPABASE_URL or SUPABASE_KEY in secrets")

        supabase = create_client(url, key)

        # Optional test query to validate the connection
        test = supabase.table("spr").select("*").limit(1).execute()

        print('test')
        print(test)
        return supabase

    except Exception as e:
        st.error(f"Supabase connection failed: {str(e)}")
        logger.error(f"Supabase client error: {str(e)}")
        return None


# ---------------- Data Loading with Advanced Caching ----------------
@st.cache_data(ttl=3600)
def load_registry_data(registry_name, table_name, _supabase_client):
    """Load registry data from Supabase with comprehensive error handling"""
    try:
        # This will raise an exception on failure
        response = _supabase_client.table(table_name).select("*").execute()
        data = response.data

        if not data:
            st.warning(f"No data returned from {registry_name}")
            return None

        df = pd.DataFrame(data)

        required_columns = ['STREET_NAME', 'HOUSE', 'BUILDING']
        missing_columns = [col for col in required_columns if col not in df.columns]


        if missing_columns:
            st.error(f"Missing required columns in {registry_name}: {missing_columns}")
            st.dataframe(df.head())
            return None

        logger.info(f"Loaded {len(df)} records from {registry_name}")
        return df

    except Exception as e:
        st.error(f"Error loading {registry_name} data: {str(e)}")
        logger.error(f"Data loading error for {registry_name}: {str(e)}")
        return None


# ---------------- Advanced Text Normalization ----------------


class AddressNormalizer:
    def __init__(self):
        self.aliases = {
            "Խ. ՀԱՅՐԻԿ": "ԽՐԻՄՅԱՆ ՀԱՅՐԻԿԻ",
            "ԽՐԻՄՅԱՆ ՀԱՅՐԻԿ": "ԽՐԻՄՅԱՆ ՀԱՅՐԻԿԻ",
        }

        self.armenian_suffixes = [
            r'\bԽՃՂ\.?', r'\bՃՂ\.?', r'\bՓ\.?', r'\bՊՈՂ\.?', r'\bԱՎ\.?', r'\bՃԱՄԲ\.?', r'\bԹԵԼԱ\.?'
        ]

        # Normalize old and new keys in this mapping
        self.old_to_new_map = {
            self._norm("Ֆրունզեի"): self._norm("Լ. Մադոյան"),
            self._norm("Լենինգրադյան"): self._norm("Վ. Սարգսյան"),
            self._norm("Կիրովականյան"): self._norm("Վանաձորի"),
            self._norm("Կալինինի"): self._norm("Գ. Նժդեհի"),
            self._norm("Կինգիսեպի"): self._norm("Վ. Չերազի"),
            self._norm("Պլեխանովի"): self._norm("Սահմանապահների"),
            self._norm("Շինարարների"): self._norm("Մ. Թետչերի"),
            self._norm("Կիրովի"): self._norm("Ն. Ռիժկովի"),
            self._norm("Լենինի"): self._norm("Տիգրան Մեծի"),
            self._norm("Խ. Հայրիկ"): self._norm("Խրիմյան Հայրիկի"),
            self._norm("Անի թաղամաս Մ. Ավետիսյան"): self._norm("Մ. Ավետիսյան"),
            self._norm("Մարքսի"): self._norm("Պ. Ջափարիձեի"),
            self._norm("Անի թաղամաս Ա. Շահինյան"): self._norm("Ա. Շահինյան"),
            self._norm("Օղակային"): self._norm("Արևելյան շրջանցող"),
            self._norm("Ռեպինի"): self._norm("Բ. Շչերբինայի"),
            self._norm("Հեղափոխության"): self._norm("Գ. Նժդեհի"),
            self._norm("Անի թաղամաս Ե. Չարենցի"): self._norm("Ե. Չարենցի"),
            self._norm("Ղուկասյան փողոց 10-րդ"): self._norm("Յ. Վարդանյան"),
            self._norm("Ղուկասյան փողոց 15-րդ"): self._norm("Յ. Վարդանյան"),
            self._norm("Ղուկասյան փողոց 11-րդ"): self._norm("Յ. Վարդանյան"),
            self._norm("Ղուկասյան փողոց 12-րդ"): self._norm("Յ. Վարդանյան"),
            self._norm("Ղուկասյան փողոց 13-րդ"): self._norm("Յ. Վարդանյան"),
            self._norm("Ղուկասյան փողոց 14-րդ"): self._norm("Յ. Վարդանյան"),
            self._norm("Սևյան"): self._norm("Հ. Ղանդիլյան"),
            self._norm("Մուշ-2  թաղամասի փողոցներից մեկը"): self._norm("Կ. Հալաբյան"),
            self._norm("Ղուկասյան"): self._norm("Յ. Վարդանյան"),
            self._norm("Խաղաղության"): self._norm("Բագրատունյաց"),
            self._norm("Մարքսի"): self._norm("Ջիվանու"),
            self._norm("Ազիզբեկովի"): self._norm("Ն. Շնորհալու"),
            self._norm("Էլեկտրո պրիբորնի 6-րդ շարք"): self._norm("Ա. Արմենյան փողոց"),
            self._norm("Էլեկտրո պրիբորնի 10-րդ շարք"): self._norm("Ա. Գևորգյան փողոց"),
            self._norm("Կիրովաբադյան փողոց"): self._norm("Ա. Թամանյան փողոց"),
            self._norm("50 ամյակի անվան փողոց"): self._norm("Ա. Մանուկյան փողոց"),
            self._norm("<<Անի>> թաղամաս 3-րդ փողոց"): self._norm("Ա. Շահինյան փողոց"),
            self._norm("Հնոցավան 2-րդ շարք"): self._norm("Ա. Պետրոսյան փողոց"),
            self._norm("Կոմսոմոլի փողոց"): self._norm("Ա. Վասիլյան փողոց"),
            self._norm("Կեցխովելի փողոց"): self._norm("Արտակ եպիսկոպոս Սմբատյան փողոց"),
            self._norm("Արվելաձե փողոց"): self._norm("Գարեգին Ա-ի փողոց"),
            self._norm("Էլեկտրո պրիբորնի 8-րդ շարք"): self._norm("Թ. Մանդալյան փողոց"),
            self._norm("Պողպատավան 3-րդ շարք"): self._norm("Ժ. Բ. Բարոնյան փողոց"),
            self._norm("Կրուպսկայա փողոց"): self._norm("Խ. Դաշտենցի փողոց"),
            self._norm("Քութաիսյան փողոց"): self._norm("Կ. Դեմիրճյան փողոց"),
            self._norm("Պողպատավան 2-րդ շարք"): self._norm("Կ. Խաչատրյան փողոց"),
            self._norm("Կույբիշևի փողոց"): self._norm("Հ. Մազմանյան փողոց"),
            self._norm("Պիոներական փողոց"): self._norm("Հ. Մելքոնյան փողոց"),
            self._norm("Պողպատավան 1-ին շարք"): self._norm("Հ. Պողոսյան փողոց"),
            self._norm("Պողպատավան 4-րդ շարք"): self._norm("Հ. Ռասկատլյան փողոց"),
            self._norm("Կատելնայա"): self._norm("Հնոցավանի 1-ին շարք"),
            self._norm("Պետ բարակներ"): self._norm("Ղ. Ղուկասյան փողոց"),
            self._norm("Մայիսյան փողոց"): self._norm("Մ. Մկրտչյան փողոց"),
            self._norm("Էլեկտրո պրիբորնի 7-րդ շարք"): self._norm("Մ. Սարգսյան փողոց"),
            self._norm("Սվերդլովի փողոց"): self._norm("Ն. Ղորղանյան փողոց"),
            self._norm("Աստղի հրապարակ"): self._norm("Շ. Ազնավուրի հրապարակ"),
            self._norm("Ս. Մուսայելյան փողոց"): self._norm("Շ. Ազնավուրի հրապարակ"),
            self._norm("Էլեկտրո պրիբորնի 11-րդ շարք"): self._norm("Ռ. Դանիելյան փողոց"),
            self._norm("Օրջոնիկիձեի փողոց"): self._norm("Ս. Մատնիշյան փողոց"),
            self._norm("Էնգելսի փողոց"): self._norm("Վ. Աճեմյան փողոց"),
            self._norm("Կենտրոնական հրապարակ"): self._norm("Վարդանանց հրապարակ"),
            self._norm("<<Անի>> թաղամաս 15-րդ փողոց"): self._norm("Ֆորալբերգի փողոց"),
        }

    def _norm(self, text):
        text = str(text).strip().upper()
        text = re.sub(r'[^\w\s]', '', text)
        return re.sub(r'\s+', ' ', text)

    def normalize(self, text):
        if pd.isna(text):
            return ""

        text = str(text).strip().upper()

        # Apply direct alias replacement first
        if text in self.aliases:
            text = self.aliases[text]

        # Remove suffixes
        for suffix in self.armenian_suffixes:
            text = re.sub(suffix, '', text, flags=re.IGNORECASE)

        # Remove special chars, extra spaces
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text)

        # Remove trailing "Ի" if it's the last letter of each word
        text = ' '.join([w[:-1] if w.endswith("Ի") else w for w in text.split()])

        # Normalize again and map old names to new ones
        text_norm = self._norm(text)
        return self.old_to_new_map.get(text_norm, text_norm)


# ---------------- Data Preprocessing Pipeline ----------------
@st.cache_data
def preprocess_registries(_spr_df, _cad_df):
    """Comprehensive data preprocessing pipeline"""
    normalizer = AddressNormalizer()

    def process_registry(df, registry_name):
        """Process individual registry with comprehensive normalization"""
        processed = df.copy()

        # Handle missing values
        processed['STREET_NAME'] = processed['STREET_NAME'].fillna('')
        processed['HOUSE'] = processed['HOUSE'].fillna('')
        processed['BUILDING'] = processed['BUILDING'].fillna('')

        # Normalize fields
        processed['STREET_NORM'] = processed['STREET_NAME'].apply(normalizer.normalize)
        processed['HOUSE_NORM'] = processed['HOUSE'].apply(normalizer.normalize)
        processed['BUILDING_NORM'] = processed['BUILDING'].apply(normalizer.normalize)

        # Create composite addresses
        processed['FULL_ADDRESS'] = (
                processed['STREET_NORM'] + " " +
                processed['HOUSE_NORM'] + " " +
                processed['BUILDING_NORM']
        ).str.strip()

        # Create search keys
        processed['SEARCH_KEY'] = (
                processed['STREET_NORM'] + "_" + processed['HOUSE_NORM']
        )

        # Data quality metrics
        processed['COMPLETENESS_SCORE'] = (
                                                  processed['STREET_NAME'].notna().astype(int) +
                                                  processed['HOUSE'].notna().astype(int) +
                                                  processed['BUILDING'].notna().astype(int)
                                          ) / 3

        logger.info(f"Processed {len(processed)} records for {registry_name}")
        return processed

    spr_processed = process_registry(_spr_df, "SPR")
    cad_processed = process_registry(_cad_df, "Cadastre")

    return spr_processed, cad_processed

# ---------------- Advanced Matching Engine ----------------
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

# ---------------- Data Quality Analysis ----------------
def analyze_data_quality(df, registry_name):
    """Comprehensive data quality analysis"""
    quality_metrics = {
        'total_records': len(df),
        'street_completeness': df['STREET_NAME'].notna().sum() / len(df),
        'house_completeness': df['HOUSE'].notna().sum() / len(df),
        'building_completeness': df['BUILDING'].notna().sum() / len(df),
        'unique_streets': df['STREET_NORM'].nunique(),
        'avg_completeness': df['COMPLETENESS_SCORE'].mean(),
        'duplicate_addresses': len(df) - df['FULL_ADDRESS'].nunique(),
    }

    return quality_metrics

# ---------------- Visualization Components ----------------
def create_match_quality_chart(matches_df):
    """Create match quality visualization"""
    if len(matches_df) == 0:
        return None

    # Score distribution
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Match Score Distribution', 'Match Type Distribution',
                        'Completeness Analysis', 'Matches Over Time'),
        specs=[[{"secondary_y": False}, {"type": "pie"}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )

    # Score distribution histogram
    fig.add_trace(
        go.Histogram(x=matches_df['MATCH_SCORE'], nbinsx=20, name='Score Distribution'),
        row=1, col=1
    )

    # Match type pie chart
    match_type_counts = matches_df['MATCH_TYPE'].value_counts()
    fig.add_trace(
        go.Pie(labels=match_type_counts.index, values=match_type_counts.values, name='Match Types'),
        row=1, col=2
    )

    # Completeness analysis
    fig.add_trace(
        go.Scatter(x=matches_df['COMPLETENESS_SPR'], y=matches_df['COMPLETENESS_CAD'],
                   mode='markers', name='Completeness Correlation'),
        row=2, col=1
    )

    # Matches over time (if timestamp available)
    if 'MATCH_TIMESTAMP' in matches_df.columns:
        matches_df['MATCH_HOUR'] = pd.to_datetime(matches_df['MATCH_TIMESTAMP']).dt.hour
        hourly_counts = matches_df['MATCH_HOUR'].value_counts().sort_index()
        fig.add_trace(
            go.Bar(x=hourly_counts.index, y=hourly_counts.values, name='Hourly Matches'),
            row=2, col=2
        )

    fig.update_layout(height=800, showlegend=True, title_text="Match Quality Analysis Dashboard")
    return fig

def create_data_quality_dashboard(spr_quality, cad_quality):
    """Create data quality comparison dashboard"""

    metrics = ['street_completeness', 'house_completeness', 'building_completeness', 'avg_completeness']
    spr_values = [spr_quality[m] for m in metrics]
    cad_values = [cad_quality[m] for m in metrics]

    # Convert to percentages and add the first value at the end to close the radar
    spr_values_pct = [v * 100 for v in spr_values] + [spr_values[0] * 100]
    cad_values_pct = [v * 100 for v in cad_values] + [cad_values[0] * 100]
    metrics_labels = metrics + [metrics[0]]  # Close the radar

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=spr_values_pct,
        theta=metrics_labels,
        fill='toself',
        name='SPR Registry',
        line_color='blue'
    ))

    fig.add_trace(go.Scatterpolar(
        r=cad_values_pct,
        theta=metrics_labels,
        fill='toself',
        name='Cadastre Registry',
        line_color='red'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=True,
        title="Data Quality Comparison (%)"
    )

    return fig

# ---------------- Export Functionality ----------------
def create_export_package(matches_df, spr_df, cad_df, quality_metrics):
    """Create comprehensive export package"""

    # Create zip buffer
    zip_buffer = BytesIO()

    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Add matched results
        matches_csv = matches_df.to_csv(index=False)
        zip_file.writestr('matched_addresses.csv', matches_csv)

        # Add unmatched records
        matched_spr_ids = set(matches_df['ADDRESS_ID_SPR'].unique())
        unmatched_spr = spr_df[~spr_df.get('ADDRESS_ID', spr_df.index).isin(matched_spr_ids)]
        unmatched_csv = unmatched_spr.to_csv(index=False)
        zip_file.writestr('unmatched_spr_addresses.csv', unmatched_csv)

        # Add quality report
        quality_report = json.dumps(quality_metrics, indent=2)
        zip_file.writestr('quality_report.json', quality_report)

        # Add matching statistics
        stats = {
            'total_spr_records': len(spr_df),
            'total_cad_records': len(cad_df),
            'total_matches': len(matches_df),
            'match_rate': len(matches_df) / len(spr_df) if len(spr_df) > 0 else 0,
            'match_types': matches_df['MATCH_TYPE'].value_counts().to_dict(),
            'score_statistics': {
                'mean': matches_df['MATCH_SCORE'].mean(),
                'median': matches_df['MATCH_SCORE'].median(),
                'std': matches_df['MATCH_SCORE'].std(),
                'min': matches_df['MATCH_SCORE'].min(),
                'max': matches_df['MATCH_SCORE'].max()
            }
        }
        zip_file.writestr('matching_statistics.json', json.dumps(stats, indent=2))

    zip_buffer.seek(0)
    return zip_buffer.getvalue()

# ---------------- Main Application ----------------
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏘️ Advanced Address Registry Matcher</h1>
        <p>Comprehensive address matching and mapping between SPR and Cadastre registries</p>
    </div>
    """, unsafe_allow_html=True)

    # Initialize session state
    if "stop_requested" not in st.session_state:
        st.session_state.stop_requested = False
    if "matching_results" not in st.session_state:
        st.session_state.matching_results = None
    if "quality_metrics" not in st.session_state:
        st.session_state.quality_metrics = None

    # Sidebar Configuration
    with st.sidebar:
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.header("🔧 Configuration")
        matching_method = st.selectbox(
            "Matching Strategy",
            ["Exact Only", "Fuzzy Only", "Hybrid (Exact + Fuzzy)", "Comprehensive Analysis"]
        )

        # Show performance warning for fuzzy matching
        if matching_method in ["Fuzzy Only", "Hybrid (Exact + Fuzzy)", "Comprehensive Analysis"]:
            st.warning("⚠️ **Performance Note:** Fuzzy matching can be slow, especially with large datasets. "
                       "Consider starting with a smaller record count for testing.")

        # Processing limits
        st.subheader("Processing Configuration")
        col1, col2 = st.columns(2)

        with col1:
            use_all_records = st.checkbox(
                "Process All Records",
                value=False,
                help="Process all available SPR records (ignores Max Records limit)"
            )

            if not use_all_records:
                max_records = st.number_input(
                    "Max Records to Process",
                    min_value=1,
                    max_value=1000000,
                    value=10000,
                    step=100,
                    help="Total number of SPR records to process for matching"
                )
            else:
                max_records = None

        with col2:
            chunk_size = st.number_input(
                "Chunk Size",
                min_value=10,
                max_value=5000,
                value=500,
                step=50,
                help="Number of records to process in each chunk (affects memory usage and progress updates)"
            )

            # Show chunk size recommendations
            if matching_method in ["Fuzzy Only", "Hybrid (Exact + Fuzzy)", "Comprehensive Analysis"]:
                if chunk_size > 1000:
                    st.info(
                        "💡 **Tip:** Smaller chunks (100-500) provide better progress tracking for fuzzy matching")

        # Fuzzy matching parameters
        if matching_method != "Exact Only":
            st.subheader("Fuzzy Matching Settings")
            col1, col2 = st.columns(2)

            with col1:
                threshold = st.slider(
                    "Fuzzy Match Threshold",
                    50, 100, 85,
                    help="Minimum similarity score for fuzzy matches (higher = stricter)"
                )

            with col2:
                # Performance estimate
                estimated_records = max_records if not use_all_records else len(
                    spr_processed) if 'spr_processed' in locals() else 0
                if estimated_records > 50000:
                    st.error(
                        "🚨 **High Volume Warning:** Processing >50k records with fuzzy matching may take significant time")
                elif estimated_records > 20000:
                    st.warning("⚠️ **Medium Volume:** Processing >20k records may take several minutes")
                else:
                    st.info("✅ **Good Volume:** Processing should complete reasonably quickly")

        # Advanced options
        st.subheader("Advanced Options")
        enable_logging = st.checkbox("Enable Detailed Logging", value=True)
        export_unmatched = st.checkbox("Include Unmatched Records", value=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 Data Overview", "🔍 Matching Process", "📈 Results Analysis", "📋 Quality Report"])

    with tab1:
        st.subheader("Registry Data Overview")

        # Load database connection
        supabase = get_supabase_client()
        if supabase is None:
            st.error("Cannot proceed without Supabase connection")
            st.stop()

        # Load data
        with st.spinner("Loading registry data..."):
            spr_df = load_registry_data("SPR", 'spr', supabase)
            cad_df = load_registry_data("Cadastre", 'cadastre', supabase)

        if spr_df is None or cad_df is None:
            st.error("Failed to load registry data")
            return

        # Preprocess data
        with st.spinner("Preprocessing data..."):
            spr_processed, cad_processed = preprocess_registries(spr_df, cad_df)

        # Data quality analysis
        spr_quality = analyze_data_quality(spr_processed, "SPR")
        cad_quality = analyze_data_quality(cad_processed, "Cadastre")

        # Display metrics with processing limits info
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("SPR Records", f"{spr_quality['total_records']:,}")
            st.metric("Unique Streets", f"{spr_quality['unique_streets']:,}")

        with col2:
            st.metric("Cadastre Records", f"{cad_quality['total_records']:,}")
            st.metric("Unique Streets", f"{cad_quality['unique_streets']:,}")

        with col3:
            st.metric("SPR Completeness", f"{spr_quality['avg_completeness']:.1%}")
            st.metric("Duplicates", f"{spr_quality['duplicate_addresses']:,}")

        with col4:
            st.metric("Cadastre Completeness", f"{cad_quality['avg_completeness']:.1%}")
            st.metric("Duplicates", f"{cad_quality['duplicate_addresses']:,}")

        # Processing limits info
        if 'max_records' in locals() and 'use_all_records' in locals():
            if not use_all_records and max_records:
                processing_limit = min(max_records, len(spr_processed))
                st.info(
                    f"🔢 **Processing Configuration:** {processing_limit:,} records will be processed in chunks of {chunk_size:,}")

                # Performance estimate for street fuzzy matching (much faster)
                if matching_method in ["Fuzzy Only", "Hybrid (Exact + Fuzzy)", "Comprehensive Analysis"]:
                    estimated_time = processing_limit * 0.002  # Much faster: 2ms per record
                    st.info(
                        f"⏱️ **Estimated Time:** ~{estimated_time / 60:.1f} minutes for street-only fuzzy matching")
            else:
                st.info(
                    f"🔢 **Processing Configuration:** All {len(spr_processed):,} records will be processed in chunks of {chunk_size:,}")

                # Performance warning for large datasets (updated for street-only matching)
                if matching_method in ["Fuzzy Only", "Hybrid (Exact + Fuzzy)", "Comprehensive Analysis"]:
                    if len(spr_processed) > 100000:
                        estimated_time = len(spr_processed) * 0.002
                        st.warning(
                            f"⚠️ **Performance Note:** Processing {len(spr_processed):,} records may take ~{estimated_time / 60:.1f} minutes")
                    else:
                        st.success("✅ **Performance:** Street-only fuzzy matching should be quite efficient")

        # Chunk size info
        if 'chunk_size' in locals():
            if chunk_size < 100:
                st.info("💡 **Small chunks:** More frequent progress updates, slightly slower overall")
            elif chunk_size > 1000:
                st.info("💡 **Large chunks:** Faster processing, less frequent progress updates")

        # Data quality visualization
        quality_fig = create_data_quality_dashboard(spr_quality, cad_quality)
        st.plotly_chart(quality_fig, use_container_width=True)

        # Sample data preview
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("SPR Sample Data")
            st.dataframe(
                spr_processed[['STREET_NAME', 'HOUSE', 'BUILDING', 'FULL_ADDRESS', 'COMPLETENESS_SCORE']].head(10))

        with col2:
            st.subheader("Cadastre Sample Data")
            st.dataframe(
                cad_processed[['STREET_NAME', 'HOUSE', 'BUILDING', 'FULL_ADDRESS', 'COMPLETENESS_SCORE']].head(10))

    with tab2:
        st.subheader("Address Matching Process")

        if 'spr_processed' not in locals() or 'cad_processed' not in locals():
            st.warning("Please load data in the Data Overview tab first")
            return

        # Matching controls
        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            if st.button("🚀 Start Matching Process", type="primary", use_container_width=True):
                st.session_state.stop_requested = False

                try:
                    # Determine processing configuration
                    if use_all_records:
                        processing_records = len(spr_processed)
                        matcher = AdvancedAddressMatcher(spr_processed, cad_processed)
                        st.info(
                            f"🔢 **Processing:** All {processing_records:,} SPR records in chunks of {chunk_size:,}")
                    else:
                        processing_records = min(max_records, len(spr_processed))
                        matcher = AdvancedAddressMatcher(spr_processed, cad_processed, max_records)
                        st.info(
                            f"🔢 **Processing:** {processing_records:,} of {len(spr_processed):,} SPR records in chunks of {chunk_size:,}")

                    # Performance warning for fuzzy matching
                    if matching_method in ["Fuzzy Only", "Hybrid (Exact + Fuzzy)", "Comprehensive Analysis"]:
                        if processing_records > 50000:
                            st.error(
                                "🚨 **High Volume Alert:** This may take 30+ minutes. Consider reducing record count for initial testing.")
                        elif processing_records > 20000:
                            st.warning(
                                "⚠️ **Medium Volume Alert:** This may take 10-20 minutes. Please be patient.")
                        elif processing_records > 5000:
                            st.info("ℹ️ **Processing Alert:** This may take 2-5 minutes for fuzzy matching.")

                    # Initialize progress tracking
                    overall_start_time = time.time()

                    # Create overall progress container
                    overall_progress_container = st.container()
                    with overall_progress_container:
                        st.subheader("📊 Overall Matching Progress")

                        overall_progress = st.progress(0)
                        overall_status = st.empty()

                        # Phase indicators
                        phase_cols = st.columns(3)
                        with phase_cols[0]:
                            exact_phase = st.empty()
                        with phase_cols[1]:
                            fuzzy_phase = st.empty()
                        with phase_cols[2]:
                            complete_phase = st.empty()

                    all_matches = []
                    phase_progress = 0
                    total_phases = 0

                    # Determine total phases
                    if matching_method in ["Exact Only", "Hybrid (Exact + Fuzzy)", "Comprehensive Analysis"]:
                        total_phases += 1
                    if matching_method in ["Fuzzy Only", "Hybrid (Exact + Fuzzy)", "Comprehensive Analysis"]:
                        total_phases += 1

                    # Phase 1: Exact matching
                    if matching_method in ["Exact Only", "Hybrid (Exact + Fuzzy)", "Comprehensive Analysis"]:
                        exact_phase.markdown("🔄 **Phase 1: Exact Matching** - In Progress")
                        overall_status.text(f"Phase 1/{total_phases}: Finding exact matches...")

                        exact_matches = matcher.find_exact_matches()
                        all_matches.append(exact_matches)

                        phase_progress += 1
                        overall_progress.progress(phase_progress / total_phases)
                        exact_phase.markdown("✅ **Phase 1: Exact Matching** - Complete")

                        st.success(f"✅ Phase 1 Complete: Found {len(exact_matches)} exact matches")

                    # Phase 2: Fuzzy matching
                    if matching_method in ["Fuzzy Only", "Hybrid (Exact + Fuzzy)", "Comprehensive Analysis"]:
                        exclude_ids = set()
                        if len(all_matches) > 0:
                            exclude_ids = set(all_matches[0]['ADDRESS_ID_SPR'].unique())

                        fuzzy_phase.markdown("🔄 **Phase 2: Fuzzy Matching** - In Progress")
                        overall_status.text(
                            f"Phase 2/{total_phases}: Running fuzzy matching (this may take time)...")

                        # Additional fuzzy matching warning
                        remaining_records = processing_records - len(exclude_ids)
                        if remaining_records > 10000:
                            st.warning(
                                f"⚠️ **Fuzzy Matching:** Processing {remaining_records:,} remaining records. This phase may take several minutes.")

                        fuzzy_matches = matcher.find_fuzzy_matches(
                            threshold=threshold,
                            chunk_size=chunk_size,
                            exclude_spr_ids=exclude_ids
                        )
                        all_matches.append(fuzzy_matches)

                        phase_progress += 1
                        overall_progress.progress(phase_progress / total_phases)
                        fuzzy_phase.markdown("✅ **Phase 2: Fuzzy Matching** - Complete")

                        st.success(f"✅ Phase 2 Complete: Found {len(fuzzy_matches)} fuzzy matches")

                    # Combine results
                    if all_matches:
                        final_matches = pd.concat([df for df in all_matches if not df.empty], ignore_index=True)
                    else:
                        final_matches = pd.DataFrame()

                    # Final processing
                    complete_phase.markdown("🔄 **Phase 3: Finalizing** - In Progress")
                    overall_status.text("Finalizing results and generating reports...")

                    end_time = time.time()
                    processing_time = end_time - overall_start_time

                    # Store results with processing info
                    st.session_state.matching_results = final_matches
                    st.session_state.processing_time = processing_time
                    st.session_state.quality_metrics = {
                        'spr_quality': spr_quality,
                        'cad_quality': cad_quality,
                        'processing_time': processing_time,
                        'matching_method': matching_method,
                        'processing_records': processing_records,
                        'total_spr_records': len(spr_processed),
                        'records_processed_percentage': processing_records / len(spr_processed) * 100,
                        'chunk_size': chunk_size
                    }

                    # Final updates
                    overall_progress.progress(1.0)
                    complete_phase.markdown("✅ **Phase 3: Finalizing** - Complete")
                    overall_status.text("🎉 All phases completed successfully!")



                    summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
                    with summary_col1:
                        st.metric("Processing Time", f"{processing_time:.2f}s")
                    with summary_col2:
                        st.metric("Records Processed", f"{processing_records:,}")
                    with summary_col3:
                        st.metric("Total Matches", f"{len(final_matches):,}")
                    with summary_col4:
                        match_rate = len(final_matches) / processing_records * 100 if processing_records > 0 else 0
                        st.metric("Match Rate", f"{match_rate:.1f}%")

                    # Performance summary
                    processing_speed = processing_records / processing_time if processing_time > 0 else 0
                    coverage_info = f"({processing_records:,} of {len(spr_processed):,} total records)" if processing_records < len(
                        spr_processed) else "(all records)"

                    st.info(f"""
                    🎯 **Matching Summary:**
                    - **Records Processed:** {processing_records:,} {coverage_info}
                    - **Chunk Size:** {chunk_size:,} records per chunk
                    - **Matches Found:** {len(final_matches):,}
                    - **Match Rate:** {match_rate:.1f}%
                    - **Processing Speed:** {processing_speed:.1f} records/second
                    - **Method Used:** {matching_method}
                    - **Coverage:** {processing_records / len(spr_processed) * 100:.1f}% of total SPR records
                    """)

                    # Show limitation warning if applicable
                    if processing_records < len(spr_processed):
                        st.warning(
                            f"⚠️ **Note:** Only {processing_records:,} out of {len(spr_processed):,} total SPR records were processed. "
                            f"To process all records, check 'Process All Records' option.")

                except Exception as e:
                    st.error(f"Error during matching process: {str(e)}")
                    logger.error(f"Matching process error: {str(e)}")
                    st.exception(e)  # Show full traceback in development

        with col2:
            if st.button("⏹ Stop Process", use_container_width=True):
                st.session_state.stop_requested = True
                st.warning("Stop requested...")

        with col3:
            if st.button("🔄 Reset", use_container_width=True):
                st.session_state.matching_results = None
                st.session_state.quality_metrics = None
                st.session_state.stop_requested = False
                st.rerun()

        # Display current matching progress/results
        if st.session_state.matching_results is not None:
            matches_df = st.session_state.matching_results
            quality_metrics = st.session_state.quality_metrics

            # Get processing info
            processing_records = quality_metrics.get('processing_records', len(spr_processed))
            total_records = quality_metrics.get('total_spr_records', len(spr_processed))
            coverage_pct = quality_metrics.get('records_processed_percentage', 100)

            # Quick statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Matches", len(matches_df))
            with col2:
                match_rate = len(matches_df) / processing_records * 100 if processing_records > 0 else 0
                st.metric("Match Rate", f"{match_rate:.1f}%")
            with col3:
                avg_score = matches_df['MATCH_SCORE'].mean() if len(matches_df) > 0 else 0
                st.metric("Avg Score", f"{avg_score:.1f}")
            with col4:
                processing_time = st.session_state.get('processing_time', 0)
                st.metric("Processing Time", f"{processing_time:.1f}s")

            # Processing coverage info
            if processing_records < total_records:
                st.info(
                    f"📊 **Processing Coverage:** {processing_records:,} of {total_records:,} total records ({coverage_pct:.1f}%)")
            else:
                st.info(f"📊 **Processing Coverage:** All {total_records:,} records processed (100%)")

            # Match type breakdown
            if len(matches_df) > 0:
                match_type_counts = matches_df['MATCH_TYPE'].value_counts()
                st.subheader("Match Type Distribution")

                # Create columns for each match type
                cols = st.columns(len(match_type_counts))
                for i, (match_type, count) in enumerate(match_type_counts.items()):
                    with cols[i]:
                        percentage = count / len(matches_df) * 100
                        st.metric(match_type, f"{count} ({percentage:.1f}%)")

    with tab3:
        st.subheader("Results Analysis & Visualization")

        if st.session_state.matching_results is None:
            st.info("No matching results available. Please run the matching process first.")
            return

        matches_df = st.session_state.matching_results

        if len(matches_df) == 0:
            st.warning("No matches found. Try adjusting the matching parameters.")
            return

        # Create comprehensive visualization
        quality_chart = create_match_quality_chart(matches_df)
        if quality_chart:
            st.plotly_chart(quality_chart, use_container_width=True)

        # Detailed match analysis
        st.subheader("Detailed Match Analysis")

        # Score distribution analysis
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Score Quality Categories")

            # Categorize matches by score
            excellent = matches_df[matches_df['MATCH_SCORE'] >= 95]
            good = matches_df[(matches_df['MATCH_SCORE'] >= 85) & (matches_df['MATCH_SCORE'] < 95)]
            fair = matches_df[(matches_df['MATCH_SCORE'] >= 75) & (matches_df['MATCH_SCORE'] < 85)]
            poor = matches_df[matches_df['MATCH_SCORE'] < 75]

            st.success(f"🟢 Excellent (95-100): {len(excellent)} matches")
            st.info(f"🔵 Good (85-94): {len(good)} matches")
            st.warning(f"🟡 Fair (75-84): {len(fair)} matches")
            st.error(f"🔴 Poor (<75): {len(poor)} matches")

        with col2:
            st.subheader("Top Scoring Matches")
            top_matches = matches_df.nlargest(10, 'MATCH_SCORE')[
                ['STREET_NAME_SPR', 'HOUSE_SPR', 'STREET_NAME_CAD', 'HOUSE_CAD', 'MATCH_SCORE']
            ]
            st.dataframe(top_matches, use_container_width=True)

        # Interactive match explorer
        st.subheader("Interactive Match Explorer")

        # Filters
        col1, col2, col3 = st.columns(3)

        with col1:
            score_filter = st.slider("Minimum Score", 0, 100, 70)

        with col2:
            match_type_filter = st.multiselect(
                "Match Types",
                options=matches_df['MATCH_TYPE'].unique(),
                default=matches_df['MATCH_TYPE'].unique()
            )

        with col3:
            street_filter = st.text_input("Filter by Street Name", "")

        # Apply filters
        filtered_matches = matches_df[
            (matches_df['MATCH_SCORE'] >= score_filter) &
            (matches_df['MATCH_TYPE'].isin(match_type_filter))
            ]

        if street_filter:
            filtered_matches = filtered_matches[
                filtered_matches['STREET_NAME_SPR'].str.contains(street_filter, case=False, na=False) |
                filtered_matches['STREET_NAME_CAD'].str.contains(street_filter, case=False, na=False)
                ]

        # Display filtered results
        st.subheader(f"Filtered Results ({len(filtered_matches)} matches)")

        # Pagination
        page_size = 50
        total_pages = (len(filtered_matches) - 1) // page_size + 1
        page = st.number_input("Page", min_value=1, max_value=total_pages, value=1) - 1

        start_idx = page * page_size
        end_idx = start_idx + page_size

        display_matches = filtered_matches.iloc[start_idx:end_idx]

        # Enhanced display with color coding
        def highlight_score(val):
            if val >= 95:
                return 'background-color: #d4edda'
            elif val >= 85:
                return 'background-color: #d1ecf1'
            elif val >= 75:
                return 'background-color: #fff3cd'
            else:
                return 'background-color: #f8d7da'

        styled_matches = display_matches.style.applymap(
            highlight_score, subset=['MATCH_SCORE']
        )

        st.dataframe(styled_matches, use_container_width=True)

        # Manual review interface
        st.subheader("Manual Review Interface")

        if len(filtered_matches) > 0:
            review_idx = st.selectbox(
                "Select match to review",
                range(len(filtered_matches)),
                format_func=lambda
                    x: f"Match {x + 1}: {filtered_matches.iloc[x]['STREET_NAME_SPR']} → {filtered_matches.iloc[x]['STREET_NAME_CAD']} (Score: {filtered_matches.iloc[x]['MATCH_SCORE']})"
            )

            if review_idx is not None:
                review_match = filtered_matches.iloc[review_idx]

                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("SPR Record")
                    st.write(f"**Street:** {review_match['STREET_NAME_SPR']}")
                    st.write(f"**House:** {review_match['HOUSE_SPR']}")
                    st.write(f"**Building:** {review_match['BUILDING_SPR']}")
                    st.write(f"**Full Address:** {review_match['FULL_ADDRESS_SPR']}")

                with col2:
                    st.subheader("Cadastre Record")
                    st.write(f"**Street:** {review_match['STREET_NAME_CAD']}")
                    st.write(f"**House:** {review_match['HOUSE_CAD']}")
                    st.write(f"**Building:** {review_match['BUILDING_CAD']}")
                    st.write(f"**Full Address:** {review_match['FULL_ADDRESS_CAD']}")

                # Match details
                st.subheader("Match Details")
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Match Score", f"{review_match['MATCH_SCORE']:.1f}")

                with col2:
                    st.metric("Match Type", review_match['MATCH_TYPE'])

                with col3:
                    st.metric("Candidates Evaluated", review_match.get('CANDIDATES_COUNT', 'N/A'))

                # Manual confirmation
                col1, col2, col3 = st.columns(3)

                with col1:
                    if st.button("✅ Confirm Match", key=f"confirm_{review_idx}"):
                        st.success("Match confirmed!")

                with col2:
                    if st.button("❌ Reject Match", key=f"reject_{review_idx}"):
                        st.error("Match rejected!")

                with col3:
                    if st.button("❓ Mark for Review", key=f"review_{review_idx}"):
                        st.warning("Marked for further review!")

    with tab4:
        st.subheader("Quality Report & Export")

        if st.session_state.matching_results is None:
            st.info("No matching results available. Please run the matching process first.")
            return

        matches_df = st.session_state.matching_results
        quality_metrics = st.session_state.quality_metrics

        # Comprehensive quality report
        st.subheader("📊 Matching Performance Report")

        # Executive summary
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Executive Summary")
            total_spr = len(spr_processed)
            total_matches = len(matches_df)
            match_rate = total_matches / total_spr if total_spr > 0 else 0

            st.write(f"**Total SPR Records:** {total_spr:,}")
            st.write(f"**Total Matches Found:** {total_matches:,}")
            st.write(f"**Overall Match Rate:** {match_rate:.1%}")
            st.write(f"**Processing Time:** {quality_metrics.get('processing_time', 0):.2f} seconds")
            st.write(f"**Matching Method:** {quality_metrics.get('matching_method', 'Unknown')}")

        with col2:
            st.markdown("### Quality Indicators")
            if len(matches_df) > 0:
                avg_score = matches_df['MATCH_SCORE'].mean()
                high_quality = len(matches_df[matches_df['MATCH_SCORE'] >= 90])
                low_quality = len(matches_df[matches_df['MATCH_SCORE'] < 80])

                st.write(f"**Average Match Score:** {avg_score:.1f}")
                st.write(
                    f"**High Quality Matches (≥90):** {high_quality} ({high_quality / len(matches_df) * 100:.1f}%)")
                st.write(
                    f"**Low Quality Matches (<80):** {low_quality} ({low_quality / len(matches_df) * 100:.1f}%)")

                # Quality assessment
                if avg_score >= 90:
                    st.success("🟢 Excellent matching quality")
                elif avg_score >= 80:
                    st.info("🔵 Good matching quality")
                elif avg_score >= 70:
                    st.warning("🟡 Fair matching quality")
                else:
                    st.error("🔴 Poor matching quality - review parameters")

        # Detailed statistics
        st.subheader("📈 Detailed Statistics")

        if len(matches_df) > 0:
            # Score statistics
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("#### Score Distribution")
                st.write(f"**Mean:** {matches_df['MATCH_SCORE'].mean():.2f}")
                st.write(f"**Median:** {matches_df['MATCH_SCORE'].median():.2f}")
                st.write(f"**Std Dev:** {matches_df['MATCH_SCORE'].std():.2f}")
                st.write(f"**Min:** {matches_df['MATCH_SCORE'].min():.2f}")
                st.write(f"**Max:** {matches_df['MATCH_SCORE'].max():.2f}")

            with col2:
                st.markdown("#### Match Type Analysis")
                match_type_stats = matches_df['MATCH_TYPE'].value_counts()
                for match_type, count in match_type_stats.items():
                    percentage = count / len(matches_df) * 100
                    st.write(f"**{match_type}:** {count} ({percentage:.1f}%)")

            with col3:
                st.markdown("#### Data Completeness")
                st.write(f"**Avg SPR Completeness:** {matches_df['COMPLETENESS_SPR'].mean():.1%}")
                st.write(f"**Avg CAD Completeness:** {matches_df['COMPLETENESS_CAD'].mean():.1%}")

                # Completeness correlation
                correlation = matches_df['COMPLETENESS_SPR'].corr(matches_df['COMPLETENESS_CAD'])
                st.write(f"**Completeness Correlation:** {correlation:.3f}")

        # Unmatched records analysis
        st.subheader("🔍 Unmatched Records Analysis")

        matched_spr_ids = set(matches_df['ADDRESS_ID_SPR'].unique()) if len(matches_df) > 0 else set()
        unmatched_spr = spr_processed[~spr_processed.get('ADDRESS_ID', spr_processed.index).isin(matched_spr_ids)]

        col1, col2 = st.columns(2)

        with col1:
            st.write(f"**Total Unmatched SPR Records:** {len(unmatched_spr)}")
            st.write(f"**Unmatched Rate:** {len(unmatched_spr) / len(spr_processed) * 100:.1f}%")

            if len(unmatched_spr) > 0:
                st.write("**Common Issues in Unmatched Records:**")
                low_completeness = len(unmatched_spr[unmatched_spr['COMPLETENESS_SCORE'] < 0.5])
                st.write(f"- Low completeness: {low_completeness} records")

                empty_streets = len(unmatched_spr[unmatched_spr['STREET_NORM'] == ''])
                st.write(f"- Empty street names: {empty_streets} records")

                empty_houses = len(unmatched_spr[unmatched_spr['HOUSE_NORM'] == ''])
                st.write(f"- Empty house numbers: {empty_houses} records")

        with col2:
            if len(unmatched_spr) > 0:
                st.write("**Sample Unmatched Records:**")
                sample_unmatched = unmatched_spr[['STREET_NAME', 'HOUSE', 'BUILDING', 'COMPLETENESS_SCORE']].head(
                    10)
                st.dataframe(sample_unmatched, use_container_width=True)

        # Export section
        st.subheader("📥 Export Options")

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("📊 Export Matched Records"):
                csv_data = matches_df.to_csv(index=False)
                st.download_button(
                    label="Download Matched Records CSV",
                    data=csv_data,
                    file_name=f"matched_addresses_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

        with col2:
            if st.button("📋 Export Unmatched Records") and len(unmatched_spr) > 0:
                unmatched_csv = unmatched_spr.to_csv(index=False)
                st.download_button(
                    label="Download Unmatched Records CSV",
                    data=unmatched_csv,
                    file_name=f"unmatched_addresses_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

        with col3:
            if st.button("📦 Export Complete Package"):
                # Create comprehensive export package
                export_data = create_export_package(matches_df, spr_processed, cad_processed, quality_metrics)
                st.download_button(
                    label="Download Complete Package",
                    data=export_data,
                    file_name=f"address_matching_package_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                    mime="application/zip"
                )

        # Recommendations
        st.subheader("💡 Recommendations")

        recommendations = []

        if len(matches_df) > 0:
            avg_score = matches_df['MATCH_SCORE'].mean()
            if avg_score < 85:
                recommendations.append("Consider lowering the matching threshold to capture more potential matches")

            low_quality_matches = len(matches_df[matches_df['MATCH_SCORE'] < 80])
            if low_quality_matches > len(matches_df) * 0.2:
                recommendations.append(
                    "High number of low-quality matches - review and possibly adjust matching parameters")

        match_rate = len(matches_df) / len(spr_processed) if len(spr_processed) > 0 else 0
        if match_rate < 0.5:
            recommendations.append(
                "Low match rate - consider data quality improvements or relaxed matching criteria")

        if len(unmatched_spr) > 0:
            low_completeness_unmatched = len(unmatched_spr[unmatched_spr['COMPLETENESS_SCORE'] < 0.5])
            if low_completeness_unmatched > len(unmatched_spr) * 0.3:
                recommendations.append(
                    "Many unmatched records have low completeness - focus on data quality improvement")

        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                st.write(f"{i}. {rec}")
        else:
            st.success("No specific recommendations - matching performance looks good!")

        # Generate summary report
        st.subheader("📝 Summary Report")

        # Calculate values outside f-string to avoid formatting issues
        avg_score = matches_df['MATCH_SCORE'].mean() if len(matches_df) > 0 else 0
        avg_score_text = f"{avg_score:.1f}" if len(matches_df) > 0 else "N/A"
        high_quality_count = len(matches_df[matches_df['MATCH_SCORE'] >= 90]) if len(matches_df) > 0 else 0
        medium_quality_count = len(
            matches_df[(matches_df['MATCH_SCORE'] >= 80) & (matches_df['MATCH_SCORE'] < 90)]) if len(
            matches_df) > 0 else 0
        low_quality_count = len(matches_df[matches_df['MATCH_SCORE'] < 80]) if len(matches_df) > 0 else 0

        summary_text = f"""
        # Address Matching Summary Report

        **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        **Method:** {quality_metrics.get('matching_method', 'Unknown')}
        **Processing Time:** {quality_metrics.get('processing_time', 0):.2f} seconds

        ## Results Overview
        - **Total SPR Records:** {len(spr_processed):,}
        - **Total Matches:** {len(matches_df):,}
        - **Match Rate:** {match_rate:.1%}
        - **Average Score:** {avg_score_text}

        ## Quality Assessment
        - **High Quality Matches (≥90):** {high_quality_count}
        - **Medium Quality Matches (80-89):** {medium_quality_count}
        - **Low Quality Matches (<80):** {low_quality_count}

        ## Recommendations
        {chr(10).join(f"- {rec}" for rec in recommendations) if recommendations else "- No specific recommendations"}
        """

        st.markdown(summary_text)

        # Export summary report
        st.download_button(
            label="📄 Download Summary Report",
            data=summary_text,
            file_name=f"matching_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown"
        )

# Run the application
if __name__ == "__main__":
    main()
