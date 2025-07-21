import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import streamlit as st
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class SPRDuplicateDetector:
    """
    Model for detecting and analyzing duplicate addresses in SPR registry
    """
    
    def __init__(self, spr_df: pd.DataFrame):
        self.spr_df = spr_df.copy()
        self.duplicate_groups = {}
        self.duplicate_stats = {}
        self.processed_df = None
        
    def detect_duplicates(self) -> Dict:
        """
        Detect duplicate addresses based on full address comparison
        Returns dictionary with duplicate analysis results
        """
        logger.info("Starting SPR duplicate detection...")
        
        # Create full address if not exists
        if 'FULL_ADDRESS' not in self.spr_df.columns:
            self.spr_df['FULL_ADDRESS'] = (
                self.spr_df.get('STREET_NAME', '').fillna('').astype(str) + " " +
                self.spr_df.get('HOUSE', '').fillna('').astype(str) + " " +
                self.spr_df.get('BUILDING', '').fillna('').astype(str)
            ).str.strip()
        
        # Filter out empty addresses
        valid_addresses = self.spr_df[self.spr_df['FULL_ADDRESS'].str.strip() != '']
        
        # Group by full address
        address_groups = valid_addresses.groupby('FULL_ADDRESS')
        
        # Find duplicates (groups with more than 1 record)
        duplicate_groups = {}
        duplicate_records = []
        
        for address, group in address_groups:
            if len(group) > 1:
                duplicate_groups[address] = {
                    'count': len(group),
                    'records': group.to_dict('records'),
                    'indices': group.index.tolist()
                }
                
                # Add duplicate info to each record
                for idx, record in group.iterrows():
                    record_dict = record.to_dict()
                    record_dict['DUPLICATE_COUNT'] = len(group)
                    record_dict['DUPLICATE_GROUP_ID'] = address
                    record_dict['RECORD_INDEX'] = idx
                    duplicate_records.append(record_dict)
        
        # Create processed dataframe with duplicate info
        self.processed_df = pd.DataFrame(duplicate_records) if duplicate_records else pd.DataFrame()
        
        # Calculate statistics
        total_records = len(valid_addresses)
        total_duplicates = len(duplicate_records)
        unique_duplicate_groups = len(duplicate_groups)
        
        self.duplicate_stats = {
            'total_records': total_records,
            'total_duplicates': total_duplicates,
            'unique_duplicate_groups': unique_duplicate_groups,
            'duplicate_rate': total_duplicates / total_records if total_records > 0 else 0,
            'empty_addresses': len(self.spr_df) - total_records,
            'largest_duplicate_group': max([group['count'] for group in duplicate_groups.values()]) if duplicate_groups else 0
        }
        
        self.duplicate_groups = duplicate_groups

        
        return {
            'duplicate_groups': duplicate_groups,
            'duplicate_stats': self.duplicate_stats,
            'processed_df': self.processed_df
        }
    
    def get_duplicate_summary(self) -> Dict:
        """Get summary statistics of duplicates"""
        return self.duplicate_stats
    
    def get_duplicate_groups_by_count(self, min_count: int = 2) -> Dict:
        """Get duplicate groups filtered by minimum count"""
        return {
            address: group for address, group in self.duplicate_groups.items()
            if group['count'] >= min_count
        }
    
    def get_top_duplicate_groups(self, top_n: int = 10) -> Dict:
        """Get top N duplicate groups by count"""
        sorted_groups = sorted(
            self.duplicate_groups.items(),
            key=lambda x: x[1]['count'],
            reverse=True
        )
        return dict(sorted_groups[:top_n])
    
    def get_duplicate_records_df(self) -> pd.DataFrame:
        """Get all duplicate records as DataFrame"""
        return self.processed_df if self.processed_df is not None else pd.DataFrame()
    
    def analyze_duplicate_patterns(self) -> Dict:
        """Analyze patterns in duplicate addresses"""
        if not self.duplicate_groups:
            return {}
        
        # Analyze by duplicate count
        count_distribution = Counter([group['count'] for group in self.duplicate_groups.values()])
        
        # Analyze by street name
        street_duplicates = defaultdict(int)
        for address, group in self.duplicate_groups.items():
            for record in group['records']:
                street_name = record.get('STREET_NAME', '')
                if street_name:
                    street_duplicates[street_name] += group['count']
        
        # Top streets with most duplicates
        top_streets = dict(sorted(street_duplicates.items(), key=lambda x: x[1], reverse=True)[:10])
        
        # Analyze by house number patterns
        house_patterns = defaultdict(int)
        for address, group in self.duplicate_groups.items():
            for record in group['records']:
                house = str(record.get('HOUSE', '')).strip()
                if house:
                    house_patterns[house] += 1
        
        top_house_patterns = dict(sorted(house_patterns.items(), key=lambda x: x[1], reverse=True)[:10])
        
        return {
            'count_distribution': dict(count_distribution),
            'top_streets_with_duplicates': top_streets,
            'top_house_patterns': top_house_patterns,
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def get_duplicate_resolution_suggestions(self) -> List[Dict]:
        """Get suggestions for resolving duplicates"""
        suggestions = []
        
        if not self.duplicate_groups:
            return suggestions
        
        for address, group in self.duplicate_groups.items():
            records = group['records']
            
            # Check if records are identical
            if len(records) > 1:
                # Compare all fields except index
                fields_to_compare = [col for col in records[0].keys() if col not in ['RECORD_INDEX', 'DUPLICATE_COUNT', 'DUPLICATE_GROUP_ID']]
                
                identical_records = True
                for field in fields_to_compare:
                    values = [str(record.get(field, '')) for record in records]
                    if len(set(values)) > 1:
                        identical_records = False
                        break
                
                suggestion = {
                    'address': address,
                    'duplicate_count': group['count'],
                    'suggestion_type': 'merge' if identical_records else 'review',
                    'suggestion_text': 'Merge identical records' if identical_records else 'Review and consolidate different records',
                    'records': records
                }
                
                suggestions.append(suggestion)
        
        return suggestions



class DuplicateDataProcessor:
    """
    Utility class for processing duplicate data for display
    """
    
    @staticmethod
    def format_duplicate_groups_for_display(duplicate_groups: Dict) -> pd.DataFrame:
        """Format duplicate groups for tabular display"""
        display_data = []
        
        for address, group in duplicate_groups.items():
            for i, record in enumerate(group['records']):
                display_data.append({
                    'Full_Address': address,
                    'Group_Size': group['count'],
                    'Record_Number': i + 1,
                    'Street_Name': record.get('STREET_NAME', ''),
                    'House': record.get('HOUSE', ''),
                    'Building': record.get('BUILDING', ''),
                    'Address_ID': record.get('ADDRESS_ID', ''),
                    'Record_Index': record.get('RECORD_INDEX', '')
                })
        
        return pd.DataFrame(display_data)
    
    @staticmethod
    def create_duplicate_summary_df(duplicate_stats: Dict) -> pd.DataFrame:
        """Create summary DataFrame for display"""
        summary_data = [
            {'Metric': 'Total Records', 'Value': duplicate_stats.get('total_records', 0)},
            {'Metric': 'Total Duplicates', 'Value': duplicate_stats.get('total_duplicates', 0)},
            {'Metric': 'Unique Duplicate Groups', 'Value': duplicate_stats.get('unique_duplicate_groups', 0)},
            {'Metric': 'Duplicate Rate', 'Value': f"{duplicate_stats.get('duplicate_rate', 0):.1%}"},
            {'Metric': 'Empty Addresses', 'Value': duplicate_stats.get('empty_addresses', 0)},
            {'Metric': 'Largest Duplicate Group', 'Value': duplicate_stats.get('largest_duplicate_group', 0)}
        ]
        
        return pd.DataFrame(summary_data)
    
    @staticmethod
    def filter_duplicates_by_criteria(
        duplicate_groups: Dict,
        min_count: int = 2,
        max_count: int = None,
        street_filter: str = None
    ) -> Dict:
        """Filter duplicate groups by various criteria"""
        filtered_groups = {}
        
        for address, group in duplicate_groups.items():
            # Filter by count
            if group['count'] < min_count:
                continue
            if max_count and group['count'] > max_count:
                continue
            
            # Filter by street name
            if street_filter:
                has_street = any(
                    street_filter.lower() in record.get('STREET_NAME', '').lower()
                    for record in group['records']
                )
                if not has_street:
                    continue
            
            filtered_groups[address] = group
        
        return filtered_groups