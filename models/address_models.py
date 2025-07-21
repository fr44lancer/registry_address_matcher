import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import re
import logging
from datetime import datetime
# Removed rapidfuzz import - not needed for exact matching only
import streamlit as st
import time




class AddressNormalizer:
    def __init__(self):
        self.aliases = {
            "Խ. ՀԱՅՐԻԿ": "ԽՐԻՄՅԱՆ ՀԱՅՐԻԿԻ",
            "ԽՐԻՄՅԱՆ ՀԱՅՐԻԿ": "ԽՐԻՄՅԱՆ ՀԱՅՐԻԿԻ",
        }

        self.armenian_suffixes = [
            r'\bԽՃՂ\.?',
            r'\bԽՃ\.?',
            r'\bրդ\.?',
            r'\bին\.?',
            r'\bՆՐԲ\.?',
            r'\bնրբ\.?',
            r'\bՆրբ\.?',
            r'\bՇԱՐՔ\.?',
            r'\bՇարք\.?',
            r'\bշարք\.?',
            r'\Անցղ\.?',
            r'\bԱՆՑՂ\.?',
            r'\bՓկղ\.?',
            r'\bՓԿՂ\.?',
            r'\bԱՆՑՂ\.?',
            r'\bշարք\.?',
            r'\bԲՆ\.?',
            r'\bՃՂ\.?',
            r'\bՓ\.?',
            r'\bՊՈՂ\.?',
            r'\bՊ\.?',
            r'\bԱՎ\.?',
            r'\bՃԱՄԲ\.?',
            r'\bՏՈՒՆ\.?',
            r'\bՏ․\.?',
            r'\b\.?',
            r'\bՀող\.?',
            r'\bՓԱԿ\.?',
            r'\bշարք\.?',
            r'\bթղմ\.?',
            r'\bԹՂՄ\.?',
            r'\bթաղամասի\.?',
            r'\bԹԵԼԱ\.?',
            r'\bՃԱՆ\.?',
            r'\bՇրջ\.?',
            r'\bթ/ղ\.?',
            r'\թղմ',
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
            self._norm("Էլեկտրո պրիբորնի 6-րդ շարք"): self._norm("Ա. Արմենյան"),
            self._norm("Էլեկտրո պրիբորնի 10-րդ շարք"): self._norm("Ա. Գևորգյան"),
            self._norm("Կիրովաբադյան փողոց"): self._norm("Ա. Թամանյան"),
            self._norm("50 ամյակի անվան փողոց"): self._norm("Ա. Մանուկյան"),
            self._norm("<<Անի>> թաղամաս 3-րդ փողոց"): self._norm("Ա. Շահինյան"),
            self._norm("Հնոցավան 2-րդ շարք"): self._norm("Ա. Պետրոսյան"),
            self._norm("Կոմսոմոլի փողոց"): self._norm("Ա. Վասիլյան փողոց"),
            self._norm("Կեցխովելի փողոց"): self._norm("Արտակ եպիսկոպոս Սմբատյան"),
            self._norm("Արվելաձե փողոց"): self._norm("Գարեգին Ա"),
            self._norm("Էլեկտրո պրիբորնի 8-րդ շարք"): self._norm("Թ. Մանդալյան"),
            self._norm("Պողպատավան 3-րդ շարք"): self._norm("Ժ. Բ. Բարոնյան"),
            self._norm("Կրուպսկայա փողոց"): self._norm("Խ. Դաշտենցի"),
            self._norm("Քութաիսյան փողոց"): self._norm("Կ. Դեմիրճյան"),
            self._norm("Պողպատավան 2-րդ շարք"): self._norm("Կ. Խաչատրյան"),
            self._norm("Կույբիշևի փողոց"): self._norm("Հ. Մազմանյան"),
            self._norm("Պիոներական փողոց"): self._norm("Հ. Մելքոնյան"),
            self._norm("Պողպատավան 1-ին շարք"): self._norm("Հ. Պողոսյան"),
            self._norm("Պողպատավան 4-րդ շարք"): self._norm("Հ. Ռասկատլյան"),
            self._norm("Կատելնայա"): self._norm("Հնոցավանի 1-ին շարք"),
            self._norm("Պետ բարակներ"): self._norm("Ղ. Ղուկասյան փողոց"),
            self._norm("Մայիսյան փողոց"): self._norm("Մ. Մկրտչյան փողոց"),
            self._norm("Էլեկտրո պրիբորնի 7-րդ շարք"): self._norm("Մ. Սարգսյան"),
            self._norm("Սվերդլովի փողոց"): self._norm("Ն. Ղորղանյան փողոց"),
            self._norm("Աստղի հրապարակ"): self._norm("Շ. Ազնավուր"),
            self._norm("Ս. Մուսայելյան փողոց"): self._norm("Շ. Ազնավուր"),
            self._norm("Էլեկտրո պրիբորնի 11-րդ շարք"): self._norm("Ռ. Դանիելյան"),
            self._norm("Օրջոնիկիձեի փողոց"): self._norm("Ս. Մատնիշյան"),
            self._norm("Էնգելսի փողոց"): self._norm("Վ. Աճեմյան"),
            self._norm("Կենտրոնական հրապարակ"): self._norm("Վարդանանց"),
            self._norm("<<Անի>> թաղամաս 15-րդ փողոց"): self._norm("Ֆորալբերգի"),
        }

    def _norm(self, text):
        text = str(text).strip()

        # Normalize Armenian "and" BEFORE upper() to handle both cases
        # Convert all forms to the ligature և for standardization
        text = text.replace('ԵՎ', 'և')
        text = text.replace('եվ', 'և')
        text = text.replace('Անտառավան', 'Անտառ')
        text = text.replace('ԱՆՏԱՌԱՎԱՆ', 'ԱՆՏԱՌ')
        text = text.replace('ՀՐ․', 'ՀՐԱՊԱՐԱԿ')
        text = re.sub(r'ՄՈՒՇ[-֊‐‑‒–—―]2', 'ՄՈՒՇ 2', text)

        # Now convert to uppercase (which will preserve the և ligature)
        text = text.upper()
        
        # Normalize Armenian ordinal numbers to Arabic numerals
        # Patterns: 1-ին, 2-րդ, 3-րդ, etc. (handles various dash types)
        text = re.sub(r'(\d+)[-֊‐‑‒–—―]?(ՐԴ|ՆԴ|ԻՆ|ԱՄ)\.?', r'\1', text, flags=re.IGNORECASE)
        
        # Remove word ՄԱՐՇԱԼ (case-insensitive)
        text = re.sub(r'\bՄԱՐՇԱԼ\b', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\bՇԱՐԼ\b', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\bՇԱՌԼ\b', '', text, flags=re.IGNORECASE)

        # Remove one-letter Armenian words (with or without dot, case-insensitive)
        text = re.sub(r'(?<!\S)[Ա-Ֆ]\.?(?!\S)', '', text, flags=re.IGNORECASE)
        
        # Remove suffixes
        for suffix in self.armenian_suffixes:
            text = re.sub(suffix, '', text, flags=re.IGNORECASE)
        
        # Strip unwanted punctuation (preserve / and -)
        text = re.sub(r'[^\w\s/\-]', '', text)
        
        # Remove trailing Ի if present
        text = ' '.join([w[:-1] if w.endswith("Ի") else w for w in text.split()])
        text = ' '.join([w[:-2] if w.endswith("ՅԻ") else w for w in text.split()])
        text = ' '.join([w[:-2] if w.endswith("ՈՒ") else w for w in text.split()])

        return re.sub(r'\s+', ' ', text).strip()

    def normalize_number_part(self, text):
        if pd.isna(text):
            return ""
        text = str(text).strip().upper()
        
        # Normalize Armenian ordinal numbers in house/building numbers
        # Patterns: 1-ին, 2-րդ, 3-րդ, etc. (handles various dash types)
        text = re.sub(r'(\d+)[-֊‐‑‒–—―]?(ՐԴ|ՆԴ|ԻՆ|ԱՄ)\.?', r'\1', text, flags=re.IGNORECASE)
        
        return re.sub(r'[^\w/\-]', '', text)

    def normalize(self, text):
        if pd.isna(text):
            return ""

        text = str(text).strip().upper()

        # Apply direct alias replacement first
        if text in self.aliases:
            text = self.aliases[text]

        # Apply normalization
        text_norm = self._norm(text)
        
        # Apply final mapping
        return self.old_to_new_map.get(text_norm, text_norm)


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


    def find_exact_matches(self):
        """Find exact matches using full address matching with detailed progress"""
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

        # Full address exact match
        status_text.text("🔍 Finding exact address matches...")
        cad_full_lookup = {row['FULL_ADDRESS']: idx for idx, row in self.cad_df.iterrows()}

        chunk_size = 500

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
                f"Processed {processed:,}/{total_spr:,} records | Found {chunk_matches} matches in current chunk")

            processed_metric.metric("Processed", f"{processed:,}/{total_spr:,}")
            matches_metric.metric("Exact Matches", f"{len(matches):,}")
            rate_metric.metric("Rate", f"{rate:.1f} records/sec" if rate > 0 else "Calculating...")
            eta_metric.metric("ETA", f"{eta:.1f}s" if eta > 0 else "Calculating...")

        # Final update
        total_time = time.time() - start_time
        progress_bar.progress(1.0)
        status_text.text(f"✅ Exact matching completed! Found {len(matches):,} matches in {total_time:.1f}s")

        return pd.DataFrame(matches)


    def _construct_original_full_address(self, row, registry_type):
        """Construct original full address from non-normalized components"""
        street = str(row.get("STREET_NAME", "")).strip()
        sub_street = str(row.get("SUB_STREET_NAME", "")).strip() if registry_type == "CAD" else ""
        house = str(row.get("HOUSE", "")).strip()
        building = str(row.get("BUILDING", "")).strip()
        
        # For CAD registry, we can also include additional address components if needed
        if registry_type == "CAD":
            # Get the original ADDRESS field if it exists (it contains the full formatted address)
            original_address = row.get("ADDRESS", "")
            if original_address:
                return str(original_address).strip()
        
        # Construct from components (include sub_street for Cadastre)
        if registry_type == "CAD":
            components = [comp for comp in [street, sub_street, house, building] if comp]
        else:
            components = [comp for comp in [street, house, building] if comp]
        return " ".join(components)

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
            "SUB_STREET_NAME_CAD": cad_row.get("SUB_STREET_NAME", ""),
            "HOUSE_CAD": cad_row.get("HOUSE", ""),
            "BUILDING_CAD": cad_row.get("BUILDING", ""),
            "FULL_ADDRESS_CAD": cad_row.get("FULL_ADDRESS", ""),
            # Original street names (before normalization)
            "ORIGINAL_STREET_NAME_SPR": spr_row.get("STREET_NAME", ""),
            "ORIGINAL_STREET_NAME_CAD": cad_row.get("STREET_NAME", ""),
            "ORIGINAL_SUB_STREET_NAME_CAD": cad_row.get("SUB_STREET_NAME", ""),
            # Original full addresses (constructed from original components)
            "ORIGINAL_FULL_ADDRESS_SPR": self._construct_original_full_address(spr_row, "SPR"),
            "ORIGINAL_FULL_ADDRESS_CAD": self._construct_original_full_address(cad_row, "CAD"),
            # Normalized full addresses (these are already normalized)
            "NORMALIZED_FULL_ADDRESS_SPR": spr_row.get("FULL_ADDRESS", ""),
            "NORMALIZED_FULL_ADDRESS_CAD": cad_row.get("FULL_ADDRESS", ""),
            "MATCH_SCORE": score,
            "MATCH_TYPE": match_type,
            "CANDIDATES_COUNT": candidates_count,
            "COMPLETENESS_SPR": spr_row.get("COMPLETENESS_SCORE", 0),
            "COMPLETENESS_CAD": cad_row.get("COMPLETENESS_SCORE", 0),
            "MATCH_TIMESTAMP": datetime.now().isoformat()
        }




def load_registry_data_from_csv(source: str, file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        st.success(f"{source} data loaded successfully from CSV")
        return df
    except Exception as e:
        st.error(f"Failed to load {source} data: {e}")
        return pd.DataFrame()


# @st.cache_data
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
        
        # Handle SUB_STREET_NAME for Cadastre registry
        if 'SUB_STREET_NAME' in processed.columns:
            processed['SUB_STREET_NAME'] = processed['SUB_STREET_NAME'].fillna('')
        else:
            processed['SUB_STREET_NAME'] = ''

        # Normalize fields
        processed['STREET_NORM'] = processed['STREET_NAME'].apply(normalizer.normalize)
        processed['SUB_STREET_NORM'] = processed['SUB_STREET_NAME'].apply(normalizer.normalize)
        processed['HOUSE_NORM'] = processed['HOUSE'].apply(normalizer.normalize_number_part)
        processed['BUILDING_NORM'] = processed['BUILDING'].apply(normalizer.normalize_number_part)

        # Create composite addresses (STREET + SUB_STREET + HOUSE + BUILDING)
        processed['FULL_ADDRESS'] = (
                processed['STREET_NORM'] + " " +
                processed['SUB_STREET_NORM'] + " " +
                processed['HOUSE_NORM'] + " " +
                processed['BUILDING_NORM']
        ).str.strip()
        
        # Clean up multiple spaces
        processed['FULL_ADDRESS'] = processed['FULL_ADDRESS'].str.replace(r'\s+', ' ', regex=True).str.strip()

        # Create search keys
        processed['SEARCH_KEY'] = (
                processed['STREET_NORM'] + "_" + processed['HOUSE_NORM']
        )

        # Data quality metrics (include SUB_STREET_NAME for Cadastre)
        if 'SUB_STREET_NAME' in df.columns and registry_name == "Cadastre":
            # For Cadastre: STREET_NAME + SUB_STREET_NAME + HOUSE + BUILDING
            processed['COMPLETENESS_SCORE'] = (
                processed['STREET_NAME'].notna().astype(int) +
                processed['SUB_STREET_NAME'].notna().astype(int) +
                processed['HOUSE'].notna().astype(int) +
                processed['BUILDING'].notna().astype(int)
            ) / 4
        else:
            # For SPR: STREET_NAME + HOUSE + BUILDING
            processed['COMPLETENESS_SCORE'] = (
                processed['STREET_NAME'].notna().astype(int) +
                processed['HOUSE'].notna().astype(int) +
                processed['BUILDING'].notna().astype(int)
            ) / 3

        return processed

    spr_processed = process_registry(_spr_df, "SPR")
    cad_processed = process_registry(_cad_df, "Cadastre")

    return spr_processed, cad_processed


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


def get_unmatched_street_names(spr_processed, cad_processed, matched_df=None):
    """Get unmatched street names between SPR and Cadastre registries"""
    
    # Get unique street names from both registries
    spr_streets = set(spr_processed['STREET_NORM'].dropna().unique())
    cad_streets = set(cad_processed['STREET_NORM'].dropna().unique())
    
    # Remove empty strings
    spr_streets.discard('')
    cad_streets.discard('')
    
    # Find streets in SPR but missing in Cadastre
    spr_only_streets = spr_streets - cad_streets
    
    # Find streets in Cadastre but missing in SPR
    cad_only_streets = cad_streets - spr_streets
    
    # Create dataframes for display
    spr_missing_df = pd.DataFrame({
        'STREET_NAME': sorted(list(spr_only_streets))
    }) if spr_only_streets else pd.DataFrame(columns=['STREET_NAME'])
    
    cad_missing_df = pd.DataFrame({
        'STREET_NAME': sorted(list(cad_only_streets))
    }) if cad_only_streets else pd.DataFrame(columns=['STREET_NAME'])
    
    return spr_missing_df, cad_missing_df