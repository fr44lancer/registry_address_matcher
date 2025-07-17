import logging
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict, Counter
from rapidfuzz import process, fuzz
import streamlit as st
import time

from config.settings import MATCHING_CONFIG
from .normalizer import AddressNormalizer

logger = logging.getLogger(__name__)


class AdvancedAddressMatcher:
    """Advanced address matching engine with multiple strategies"""
    
    def __init__(self, spr_df: pd.DataFrame, cad_df: pd.DataFrame, max_records: Optional[int] = None):
        self.normalizer = AddressNormalizer()
        
        # Limit SPR records if specified
        if max_records is not None and max_records < len(spr_df):
            self.spr_df = spr_df.head(max_records).copy()
            if 'st' in globals():
                st.info(f"📊 Processing limited to {max_records:,} SPR records out of {len(spr_df):,} total records")
        else:
            self.spr_df = spr_df.copy()
        
        self.cad_df = cad_df
        self.original_spr_count = len(spr_df)
        self.processing_count = len(self.spr_df)
        
        # Create indices for fast lookups
        self.street_index = self._create_street_index()
        self.house_index = self._create_house_index()
        self.search_key_index = self._create_search_key_index()
        self.house_flexible_index, self.component_index = self._create_flexible_indices()
    
    def _create_street_index(self) -> Dict[str, List[int]]:
        """Create street-based index for fast lookups"""
        index = defaultdict(list)
        for idx, row in self.cad_df.iterrows():
            street = row.get('STREET_NORM', '')
            if street:
                index[street].append(idx)
        return dict(index)
    
    def _create_house_index(self) -> Dict[str, List[int]]:
        """Create house-based index for fast lookups"""
        index = defaultdict(list)
        for idx, row in self.cad_df.iterrows():
            house = row.get('HOUSE_NORM', '')
            if house:
                index[house].append(idx)
        return dict(index)
    
    def _create_search_key_index(self) -> Dict[str, List[int]]:
        """Create search key index for fast lookups"""
        index = defaultdict(list)
        for idx, row in self.cad_df.iterrows():
            search_key = row.get('SEARCH_KEY', '')
            if search_key:
                index[search_key].append(idx)
        return dict(index)
    
    def _create_flexible_indices(self) -> Tuple[Dict[str, List[int]], Dict[str, List[int]]]:
        """Create flexible indices for enhanced matching"""
        house_flexible_index = defaultdict(list)
        component_index = defaultdict(list)
        
        for idx, row in self.cad_df.iterrows():
            # Flexible house matching (first 2 characters)
            house = str(row.get('HOUSE_NORM', ''))
            if len(house) >= 2:
                house_flexible_index[house[:2]].append(idx)
            
            # Component-based index
            street = str(row.get('STREET_NORM', ''))
            if street:
                # Index by first few characters of street
                component_index[street[:3]].append(idx)
        
        return dict(house_flexible_index), dict(component_index)
    
    def _calculate_fuzzy_score(self, str1: str, str2: str) -> float:
        """Calculate weighted fuzzy matching score"""
        if not str1 or not str2:
            return 0.0
        
        # Calculate different types of fuzzy scores
        ratio_score = fuzz.ratio(str1, str2)
        partial_ratio_score = fuzz.partial_ratio(str1, str2)
        token_sort_score = fuzz.token_sort_ratio(str1, str2)
        token_set_score = fuzz.token_set_ratio(str1, str2)
        
        # Apply weighted average
        weighted_score = (
            ratio_score * MATCHING_CONFIG.fuzzy_ratio_weight +
            partial_ratio_score * MATCHING_CONFIG.partial_ratio_weight +
            token_sort_score * MATCHING_CONFIG.token_sort_weight +
            token_set_score * MATCHING_CONFIG.token_set_weight
        )
        
        return weighted_score
    
    def _get_match_quality(self, score: float) -> str:
        """Determine match quality based on score"""
        if score >= MATCHING_CONFIG.threshold_excellent:
            return "Excellent"
        elif score >= MATCHING_CONFIG.threshold_good:
            return "Good"
        elif score >= MATCHING_CONFIG.threshold_poor:
            return "Poor"
        else:
            return "No Match"
    
    def _exact_match(self, spr_row: pd.Series) -> List[Dict[str, Any]]:
        """Perform exact matching using indices"""
        matches = []
        search_key = spr_row.get('SEARCH_KEY', '')
        
        if search_key in self.search_key_index:
            for cad_idx in self.search_key_index[search_key]:
                cad_row = self.cad_df.iloc[cad_idx]
                matches.append({
                    'spr_index': spr_row.name,
                    'cad_index': cad_idx,
                    'match_score': 100.0,
                    'match_quality': 'Excellent',
                    'match_type': 'Exact',
                    'spr_search_key': search_key,
                    'cad_search_key': cad_row.get('SEARCH_KEY', ''),
                    'spr_address': spr_row.get('FULL_ADDRESS', ''),
                    'cad_address': cad_row.get('FULL_ADDRESS', ''),
                    'spr_street_name': spr_row.get('STREET_NAME', ''),
                    'cad_street_name': cad_row.get('STREET_NAME', ''),
                    'spr_house': spr_row.get('HOUSE', ''),
                    'cad_house': cad_row.get('HOUSE', ''),
                    'spr_building': spr_row.get('BUILDING', ''),
                    'cad_building': cad_row.get('BUILDING', ''),
                    'completeness_spr': spr_row.get('COMPLETENESS_SCORE', 0),
                    'completeness_cad': cad_row.get('COMPLETENESS_SCORE', 0)
                })
        
        return matches
    
    def _fuzzy_match(self, spr_row: pd.Series) -> List[Dict[str, Any]]:
        """Perform fuzzy matching with multiple strategies"""
        matches = []
        spr_search_key = spr_row.get('SEARCH_KEY', '')
        spr_street = spr_row.get('STREET_NORM', '')
        spr_house = spr_row.get('HOUSE_NORM', '')
        
        # Strategy 1: Street-based matching
        candidates = set()
        if spr_street in self.street_index:
            candidates.update(self.street_index[spr_street])
        
        # Strategy 2: House-based matching
        if spr_house in self.house_index:
            candidates.update(self.house_index[spr_house])
        
        # Strategy 3: Flexible house matching
        if len(spr_house) >= 2:
            house_prefix = spr_house[:2]
            if house_prefix in self.house_flexible_index:
                candidates.update(self.house_flexible_index[house_prefix])
        
        # Strategy 4: Component-based matching
        if len(spr_street) >= 3:
            street_prefix = spr_street[:3]
            if street_prefix in self.component_index:
                candidates.update(self.component_index[street_prefix])
        
        # Evaluate candidates
        for cad_idx in candidates:
            cad_row = self.cad_df.iloc[cad_idx]
            cad_search_key = cad_row.get('SEARCH_KEY', '')
            
            # Calculate fuzzy score
            score = self._calculate_fuzzy_score(spr_search_key, cad_search_key)
            
            if score >= MATCHING_CONFIG.threshold_poor:
                matches.append({
                    'spr_index': spr_row.name,
                    'cad_index': cad_idx,
                    'match_score': score,
                    'match_quality': self._get_match_quality(score),
                    'match_type': 'Fuzzy',
                    'spr_search_key': spr_search_key,
                    'cad_search_key': cad_search_key,
                    'spr_address': spr_row.get('FULL_ADDRESS', ''),
                    'cad_address': cad_row.get('FULL_ADDRESS', ''),
                    'spr_street_name': spr_row.get('STREET_NAME', ''),
                    'cad_street_name': cad_row.get('STREET_NAME', ''),
                    'spr_house': spr_row.get('HOUSE', ''),
                    'cad_house': cad_row.get('HOUSE', ''),
                    'spr_building': spr_row.get('BUILDING', ''),
                    'cad_building': cad_row.get('BUILDING', ''),
                    'completeness_spr': spr_row.get('COMPLETENESS_SCORE', 0),
                    'completeness_cad': cad_row.get('COMPLETENESS_SCORE', 0)
                })
        
        # Sort by score and limit results
        matches.sort(key=lambda x: x['match_score'], reverse=True)
        return matches[:MATCHING_CONFIG.max_results]
    
    def match_addresses(self, use_progress_bar: bool = True) -> pd.DataFrame:
        """Main matching function with progress tracking"""
        all_matches = []
        
        # Progress tracking
        if use_progress_bar and 'st' in globals():
            progress_bar = st.progress(0)
            status_text = st.empty()
        
        start_time = time.time()
        
        for i, (spr_idx, spr_row) in enumerate(self.spr_df.iterrows()):
            # Update progress
            if use_progress_bar and 'st' in globals() and i % 100 == 0:
                progress = (i + 1) / len(self.spr_df)
                progress_bar.progress(progress)
                status_text.text(f"Processing record {i+1:,} of {len(self.spr_df):,}")
            
            # Try exact matching first
            matches = self._exact_match(spr_row)
            
            # If no exact matches, try fuzzy matching
            if not matches:
                matches = self._fuzzy_match(spr_row)
            
            # Add matches to results
            all_matches.extend(matches)
        
        # Complete progress bar
        if use_progress_bar and 'st' in globals():
            progress_bar.progress(1.0)
            elapsed_time = time.time() - start_time
            status_text.text(f"✅ Matching completed in {elapsed_time:.2f} seconds")
        
        # Create results DataFrame
        if all_matches:
            matches_df = pd.DataFrame(all_matches)
            logger.info(f"Found {len(matches_df)} total matches")
            return matches_df
        else:
            logger.warning("No matches found")
            return pd.DataFrame()
    
    def get_matching_statistics(self) -> Dict[str, Any]:
        """Get statistics about the matching process"""
        return {
            'total_spr_records': self.original_spr_count,
            'processed_spr_records': self.processing_count,
            'total_cad_records': len(self.cad_df),
            'street_index_size': len(self.street_index),
            'house_index_size': len(self.house_index),
            'search_key_index_size': len(self.search_key_index),
            'flexible_house_index_size': len(self.house_flexible_index),
            'component_index_size': len(self.component_index)
        }