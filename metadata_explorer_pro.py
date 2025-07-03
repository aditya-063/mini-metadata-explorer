import streamlit as st
import pandas as pd
import numpy as np
import json
import re
from typing import Dict, List, Tuple, Any, Optional
from io import StringIO
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
from datetime import datetime, timedelta
import random
from collections import defaultdict
import uuid

# Page config
st.set_page_config(
    page_title="Mini Metadata Explorer Pro",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .dataset-card {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        background-color: #f8f9fa;
    }
    .metadata-table {
        font-size: 0.9em;
    }
    .tag-pill {
        display: inline-block;
        padding: 0.2rem 0.5rem;
        margin: 0.1rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    .tag-pii { background-color: #ff6b6b; color: white; }
    .tag-category { background-color: #4ecdc4; color: white; }
    .tag-identifier { background-color: #45b7d1; color: white; }
    .tag-feature { background-color: #96ceb4; color: white; }
    .tag-custom { background-color: #feca57; color: black; }
    .tag-glossary { background-color: #a55eea; color: white; }
    .comment-box {
        border-left: 3px solid #667eea;
        padding: 0.5rem;
        margin: 0.5rem 0;
        background-color: #f8f9fa;
        border-radius: 0 5px 5px 0;
    }
    .rule-violation {
        background-color: #ffe6e6;
        border-left: 3px solid #ff4757;
        padding: 0.5rem;
        margin: 0.2rem 0;
        border-radius: 0 3px 3px 0;
    }
    .rule-passed {
        background-color: #e6ffe6;
        border-left: 3px solid #2ed573;
        padding: 0.5rem;
        margin: 0.2rem 0;
        border-radius: 0 3px 3px 0;
    }
    .footer {
        text-align: center;
        padding: 2rem 0;
        color: #666;
        border-top: 1px solid #e0e0e0;
        margin-top: 3rem;
    }
    .search-highlight {
        background-color: yellow;
        font-weight: bold;
    }
    .outlier-badge {
        background-color: #ff9f43;
        color: white;
        padding: 0.1rem 0.3rem;
        border-radius: 3px;
        font-size: 0.7rem;
    }
</style>
""", unsafe_allow_html=True)

# Sample users for collaboration simulation
SAMPLE_USERS = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank"]

class DataProfiler:
    """Enhanced data profiling with visualizations and outlier detection"""
    
    @staticmethod
    def detect_outliers_iqr(series: pd.Series) -> List[int]:
        """Detect outliers using IQR method"""
        if not pd.api.types.is_numeric_dtype(series):
            return []
        
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = series[(series < lower_bound) | (series > upper_bound)]
        return outliers.index.tolist()
    
    @staticmethod
    def detect_outliers_zscore(series: pd.Series, threshold: float = 3) -> List[int]:
        """Detect outliers using Z-score method"""
        if not pd.api.types.is_numeric_dtype(series):
            return []
        
        z_scores = np.abs((series - series.mean()) / series.std())
        outliers = series[z_scores > threshold]
        return outliers.index.tolist()
    
    @staticmethod
    def create_column_visualization(df: pd.DataFrame, column: str) -> go.Figure:
        """Create appropriate visualization for column"""
        col_data = df[column].dropna()
        
        if pd.api.types.is_numeric_dtype(col_data):
            # Histogram for numeric data
            fig = px.histogram(
                x=col_data,
                nbins=30,
                title=f"Distribution of {column}",
                labels={'x': column, 'y': 'Frequency'}
            )
            fig.update_layout(height=300)
            return fig
        else:
            # Bar chart for categorical data
            value_counts = col_data.value_counts().head(10)
            
            if len(value_counts) <= 5:
                # Pie chart for few categories
                fig = px.pie(
                    values=value_counts.values,
                    names=value_counts.index,
                    title=f"Distribution of {column}"
                )
            else:
                # Bar chart for many categories
                fig = px.bar(
                    x=value_counts.index,
                    y=value_counts.values,
                    title=f"Top 10 Values in {column}",
                    labels={'x': column, 'y': 'Count'}
                )
            
            fig.update_layout(height=300)
            return fig

class MetadataExtractor:
    """Enhanced metadata extraction with outlier detection"""
    
    @staticmethod
    def get_column_metadata(df: pd.DataFrame, column: str) -> Dict[str, Any]:
        """Extract enhanced metadata for a single column"""
        col_data = df[column]
        
        # Basic stats
        total_rows = len(df)
        missing_count = col_data.isnull().sum()
        missing_percentage = (missing_count / total_rows) * 100
        unique_count = col_data.nunique()
        
        # Sample values (non-null)
        sample_values = col_data.dropna().unique()[:5].tolist()
        
        # Data type
        dtype = str(col_data.dtype)
        
        # Outlier detection
        outliers_iqr = DataProfiler.detect_outliers_iqr(col_data)
        outliers_zscore = DataProfiler.detect_outliers_zscore(col_data)
        
        # Additional stats for numeric columns
        stats = {}
        if pd.api.types.is_numeric_dtype(col_data):
            stats = {
                'mean': col_data.mean(),
                'median': col_data.median(),
                'std': col_data.std(),
                'min': col_data.min(),
                'max': col_data.max()
            }
        
        return {
            'column_name': column,
            'data_type': dtype,
            'missing_percentage': round(missing_percentage, 2),
            'missing_count': missing_count,
            'unique_count': unique_count,
            'sample_values': sample_values,
            'total_rows': total_rows,
            'outliers_iqr': len(outliers_iqr),
            'outliers_zscore': len(outliers_zscore),
            'outlier_indices_iqr': outliers_iqr[:10],  # Limit for performance
            'outlier_indices_zscore': outliers_zscore[:10],
            'stats': stats
        }
    
    @staticmethod
    def auto_detect_category(column_name: str, metadata: Dict[str, Any]) -> str:
        """Auto-detect column category based on name and metadata"""
        column_lower = column_name.lower()
        
        # PII detection
        pii_keywords = ['email', 'name', 'phone', 'address', 'ssn', 'social', 'passport', 'license']
        if any(keyword in column_lower for keyword in pii_keywords):
            return "PII"
        
        # Identifier detection (high uniqueness)
        uniqueness_ratio = metadata['unique_count'] / metadata['total_rows']
        if uniqueness_ratio > 0.8:
            return "Identifier"
        
        # Category detection (numerical with few unique values)
        if 'int' in metadata['data_type'] or 'float' in metadata['data_type']:
            if metadata['unique_count'] < 20:  # Arbitrary threshold
                return "Category"
        
        # Default
        return "Feature"

class LineageAnalyzer:
    """Enhanced lineage analysis with graph visualization"""
    
    @staticmethod
    def find_column_matches(datasets: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        """Find potential column relationships across datasets"""
        matches = []
        dataset_names = list(datasets.keys())
        
        for i, dataset1 in enumerate(dataset_names):
            for j, dataset2 in enumerate(dataset_names[i+1:], i+1):
                df1 = datasets[dataset1]
                df2 = datasets[dataset2]
                
                # Find exact column name matches
                common_columns = set(df1.columns) & set(df2.columns)
                
                for col in common_columns:
                    match = {
                        'source_dataset': dataset1,
                        'target_dataset': dataset2,
                        'column_name': col,
                        'match_type': 'Exact Name Match',
                        'confidence': 'High',
                        'similarity_score': 1.0
                    }
                    matches.append(match)
                
                # Find similar column names
                for col1 in df1.columns:
                    for col2 in df2.columns:
                        if col1 != col2:
                            similarity = LineageAnalyzer._calculate_similarity(col1, col2)
                            if similarity > 0.6:  # Threshold for similarity
                                match = {
                                    'source_dataset': dataset1,
                                    'target_dataset': dataset2,
                                    'column_name': f"{col1} → {col2}",
                                    'match_type': 'Similar Name',
                                    'confidence': 'Medium' if similarity > 0.8 else 'Low',
                                    'similarity_score': similarity
                                }
                                matches.append(match)
        
        return matches
    
    @staticmethod
    def _calculate_similarity(col1: str, col2: str) -> float:
        """Calculate similarity score between two column names"""
        col1_clean = re.sub(r'[^a-zA-Z0-9]', '', col1.lower())
        col2_clean = re.sub(r'[^a-zA-Z0-9]', '', col2.lower())
        
        # Jaccard similarity
        set1 = set(col1_clean)
        set2 = set(col2_clean)
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    @staticmethod
    def create_lineage_graph(datasets: Dict[str, pd.DataFrame], matches: List[Dict[str, Any]]) -> go.Figure:
        """Create interactive lineage graph"""
        G = nx.Graph()
        
        # Add dataset nodes
        for dataset_name in datasets.keys():
            G.add_node(dataset_name, node_type='dataset')
        
        # Add column nodes and edges
        for match in matches:
            source = match['source_dataset']
            target = match['target_dataset']
            
            # Add edge with weight based on confidence
            weight = {'High': 3, 'Medium': 2, 'Low': 1}[match['confidence']]
            G.add_edge(source, target, 
                      weight=weight, 
                      match_info=match['column_name'],
                      match_type=match['match_type'],
                      confidence=match['confidence'])
        
        # Create layout
        pos = nx.spring_layout(G)
        
        # Extract edges
        edge_x = []
        edge_y = []
        edge_info = []
        
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
            edge_data = G.edges[edge]
            edge_info.append(f"{edge_data['match_info']}<br>Type: {edge_data['match_type']}<br>Confidence: {edge_data['confidence']}")
        
        # Create figure
        fig = go.Figure()
        
        # Add edges
        fig.add_trace(go.Scatter(x=edge_x, y=edge_y,
                                line=dict(width=2, color='#888'),
                                hoverinfo='none',
                                mode='lines',
                                showlegend=False))
        
        # Add nodes
        node_x = []
        node_y = []
        node_text = []
        node_info = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node)
            
            # Node info
            df = datasets[node]
            info = f"Dataset: {node}<br>Rows: {len(df):,}<br>Columns: {len(df.columns)}"
            node_info.append(info)
        
        fig.add_trace(go.Scatter(x=node_x, y=node_y,
                                mode='markers+text',
                                marker=dict(size=50, color='lightblue', line=dict(width=2, color='navy')),
                                text=node_text,
                                textposition="middle center",
                                hovertext=node_info,
                                hoverinfo="text",
                                showlegend=False))
        
        fig.update_layout(
            title="Dataset Lineage Graph",
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            annotations=[ dict(
                text="Hover over nodes and edges for details",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.005, y=-0.002,
                xanchor='left', yanchor='bottom',
                font=dict(color='gray', size=12)
            )],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=500
        )
        
        return fig

class ColumnImpactAnalyzer:
    """Analyze column impact across datasets"""
    
    @staticmethod
    def analyze_column_impact(datasets: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, Any]]:
        """Analyze impact of each column across datasets"""
        column_impact = defaultdict(lambda: {
            'datasets': [],
            'similar_columns': [],
            'total_appearances': 0,
            'impact_score': 0
        })
        
        # Track exact matches
        for dataset_name, df in datasets.items():
            for column in df.columns:
                column_impact[column]['datasets'].append(dataset_name)
                column_impact[column]['total_appearances'] += 1
        
        # Track similar columns
        all_columns = []
        for df in datasets.values():
            all_columns.extend(df.columns)
        
        for col1 in set(all_columns):
            for col2 in set(all_columns):
                if col1 != col2:
                    similarity = LineageAnalyzer._calculate_similarity(col1, col2)
                    if similarity > 0.6:
                        if col2 not in column_impact[col1]['similar_columns']:
                            column_impact[col1]['similar_columns'].append(col2)
        
        # Calculate impact scores
        for column, impact in column_impact.items():
            # Score based on appearances and similar columns
            impact['impact_score'] = (
                impact['total_appearances'] * 2 + 
                len(impact['similar_columns']) * 0.5
            )
        
        return dict(column_impact)

class DataQualityRules:
    """Data quality rule engine"""
    
    @staticmethod
    def validate_rules(df: pd.DataFrame, column: str, rules: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate data quality rules for a column"""
        results = []
        col_data = df[column]
        
        for rule in rules:
            rule_type = rule['type']
            passed = False
            violation_count = 0
            violations = []
            
            if rule_type == 'unique':
                duplicates = col_data.duplicated()
                passed = not duplicates.any()
                violation_count = duplicates.sum()
                violations = df[duplicates].index.tolist()[:10]
            
            elif rule_type == 'not_null':
                nulls = col_data.isnull()
                passed = not nulls.any()
                violation_count = nulls.sum()
                violations = df[nulls].index.tolist()[:10]
            
            elif rule_type == 'regex':
                pattern = rule.get('pattern', '')
                if pattern:
                    non_null_data = col_data.dropna()
                    matches = non_null_data.astype(str).str.match(pattern)
                    passed = matches.all()
                    violation_count = (~matches).sum()
                    violations = non_null_data[~matches].index.tolist()[:10]
            
            elif rule_type == 'range':
                min_val = rule.get('min')
                max_val = rule.get('max')
                if pd.api.types.is_numeric_dtype(col_data):
                    valid = True
                    if min_val is not None:
                        valid &= (col_data >= min_val).all()
                    if max_val is not None:
                        valid &= (col_data <= max_val).all()
                    passed = valid
                    
                    # Count violations
                    violations_mask = pd.Series([False] * len(col_data))
                    if min_val is not None:
                        violations_mask |= (col_data < min_val)
                    if max_val is not None:
                        violations_mask |= (col_data > max_val)
                    violation_count = violations_mask.sum()
                    violations = df[violations_mask].index.tolist()[:10]
            
            results.append({
                'rule': rule,
                'passed': passed,
                'violation_count': violation_count,
                'violations': violations,
                'total_checked': len(col_data.dropna()) if rule_type != 'not_null' else len(col_data)
            })
        
        return results

class MetadataVersioning:
    """Metadata versioning and history tracking"""
    
    @staticmethod
    def create_snapshot(metadata: Dict, annotations: Dict, version_name: str = None) -> Dict[str, Any]:
        """Create a metadata snapshot"""
        if version_name is None:
            version_name = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        return {
            'version_id': str(uuid.uuid4()),
            'version_name': version_name,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata.copy(),
            'annotations': annotations.copy()
        }
    
    @staticmethod
    def compare_versions(version1: Dict[str, Any], version2: Dict[str, Any]) -> Dict[str, List[str]]:
        """Compare two metadata versions"""
        changes = {
            'added_columns': [],
            'removed_columns': [],
            'modified_descriptions': [],
            'modified_tags': [],
            'modified_rules': []
        }
        
        # Compare annotations
        ann1 = version1.get('annotations', {})
        ann2 = version2.get('annotations', {})
        
        all_datasets = set(ann1.keys()) | set(ann2.keys())
        
        for dataset in all_datasets:
            cols1 = set(ann1.get(dataset, {}).keys())
            cols2 = set(ann2.get(dataset, {}).keys())
            
            # Added/removed columns
            changes['added_columns'].extend([f"{dataset}.{col}" for col in cols2 - cols1])
            changes['removed_columns'].extend([f"{dataset}.{col}" for col in cols1 - cols2])
            
            # Modified annotations
            common_cols = cols1 & cols2
            for col in common_cols:
                ann1_col = ann1[dataset][col]
                ann2_col = ann2[dataset][col]
                
                if ann1_col.get('description') != ann2_col.get('description'):
                    changes['modified_descriptions'].append(f"{dataset}.{col}")
                
                if set(ann1_col.get('custom_tags', [])) != set(ann2_col.get('custom_tags', [])):
                    changes['modified_tags'].append(f"{dataset}.{col}")
        
        return changes

def render_tag_pill(tag: str, tag_type: str = "custom") -> str:
    """Render a tag as a styled pill"""
    return f'<span class="tag-pill tag-{tag_type.lower()}">{tag}</span>'

def search_metadata(datasets: Dict, metadata: Dict, annotations: Dict, search_term: str) -> List[Dict[str, str]]:
    """Search across all metadata and annotations"""
    results = []
    search_lower = search_term.lower()
    
    for dataset_name in datasets.keys():
        for column_name in datasets[dataset_name].columns:
            # Search in column name
            if search_lower in column_name.lower():
                results.append({
                    'dataset': dataset_name,
                    'column': column_name,
                    'match_type': 'Column Name',
                    'match_text': column_name
                })
            
            # Search in description
            desc = annotations.get(dataset_name, {}).get(column_name, {}).get('description', '')
            if search_lower in desc.lower():
                results.append({
                    'dataset': dataset_name,
                    'column': column_name,
                    'match_type': 'Description',
                    'match_text': desc
                })
            
            # Search in tags
            tags = annotations.get(dataset_name, {}).get(column_name, {}).get('custom_tags', [])
            for tag in tags:
                if search_lower in tag.lower():
                    results.append({
                        'dataset': dataset_name,
                        'column': column_name,
                        'match_type': 'Tag',
                        'match_text': tag
                    })
    
    return results

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🔍 Mini Metadata Explorer Pro</h1>
        <p>Enterprise-grade data catalog with advanced profiling, lineage, and collaboration features</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'datasets' not in st.session_state:
        st.session_state.datasets = {}
    if 'metadata' not in st.session_state:
        st.session_state.metadata = {}
    if 'user_annotations' not in st.session_state:
        st.session_state.user_annotations = {}
    if 'glossary_terms' not in st.session_state:
        st.session_state.glossary_terms = {}
    if 'data_quality_rules' not in st.session_state:
        st.session_state.data_quality_rules = {}
    if 'metadata_versions' not in st.session_state:
        st.session_state.metadata_versions = {}
    if 'comments' not in st.session_state:
        st.session_state.comments = {}
    
    # Sidebar
    with st.sidebar:
        st.header("📁 Upload Datasets")
        uploaded_files = st.file_uploader(
            "Choose CSV files",
            type=['csv'],
            accept_multiple_files=True,
            help="Upload one or more CSV files to explore their metadata"
        )
        
        if uploaded_files:
            for uploaded_file in uploaded_files:
                if uploaded_file.name not in st.session_state.datasets:
                    try:
                        df = pd.read_csv(uploaded_file)
                        st.session_state.datasets[uploaded_file.name] = df
                        
                        # Extract metadata
                        file_metadata = {}
                        for column in df.columns:
                            col_meta = MetadataExtractor.get_column_metadata(df, column)
                            col_meta['auto_category'] = MetadataExtractor.auto_detect_category(column, col_meta)
                            file_metadata[column] = col_meta
                        
                        st.session_state.metadata[uploaded_file.name] = file_metadata
                        
                        # Initialize user annotations
                        if uploaded_file.name not in st.session_state.user_annotations:
                            st.session_state.user_annotations[uploaded_file.name] = {}
                            for column in df.columns:
                                st.session_state.user_annotations[uploaded_file.name][column] = {
                                    'description': '',
                                    'custom_tags': [],
                                    'comments': '',
                                    'glossary_term': '',
                                    'business_owner': ''
                                }
                        
                        # Initialize data quality rules
                        if uploaded_file.name not in st.session_state.data_quality_rules:
                            st.session_state.data_quality_rules[uploaded_file.name] = {}
                            for column in df.columns:
                                st.session_state.data_quality_rules[uploaded_file.name][column] = []
                        
                        # Initialize comments
                        if uploaded_file.name not in st.session_state.comments:
                            st.session_state.comments[uploaded_file.name] = {}
                            for column in df.columns:
                                st.session_state.comments[uploaded_file.name][column] = []
                        
                        # Create initial version
                        if uploaded_file.name not in st.session_state.metadata_versions:
                            snapshot = MetadataVersioning.create_snapshot(
                                st.session_state.metadata[uploaded_file.name],
                                st.session_state.user_annotations[uploaded_file.name],
                                "Initial Upload"
                            )
                            st.session_state.metadata_versions[uploaded_file.name] = [snapshot]
                        
                        st.success(f"✅ Loaded {uploaded_file.name}")
                    except Exception as e:
                        st.error(f"❌ Error loading {uploaded_file.name}: {str(e)}")
        
        # Global search
        st.header("🔍 Global Search")
        search_term = st.text_input("Search columns, descriptions, tags...")
        
        if search_term and st.session_state.datasets:
            search_results = search_metadata(
                st.session_state.datasets,
                st.session_state.metadata,
                st.session_state.user_annotations,
                search_term
            )
            
            if search_results:
                st.write(f"Found {len(search_results)} results:")
                for result in search_results[:5]:  # Show top 5
                    st.write(f"📄 **{result['dataset']}**.{result['column']}")
                    st.write(f"   *{result['match_type']}*: {result['match_text'][:50]}...")
            else:
                st.write("No results found")
        
        # Dataset overview
        if st.session_state.datasets:
            st.header("📊 Dataset Overview")
            for name, df in st.session_state.datasets.items():
                st.write(f"**{name}**")
                st.write(f"• Rows: {len(df):,}")
                st.write(f"• Columns: {len(df.columns)}")
                
                # Show data quality summary
                if name in st.session_state.data_quality_rules:
                    total_rules = sum(len(rules) for rules in st.session_state.data_quality_rules[name].values())
                    st.write(f"• DQ Rules: {total_rules}")
                
                st.write("---")
    
    # Main content
    if not st.session_state.datasets:
        st.info("👆 Upload CSV files using the sidebar to get started!")
        return
    
    # Main navigation
    main_tabs = ["📋 Data Exploration", "📚 Business Glossary", "🔗 Lineage Analysis", "📊 Column Impact", "🕘 Version History"]
    selected_tab = st.selectbox("Navigate to:", main_tabs)
    
    if selected_tab == "📋 Data Exploration":
        render_data_exploration()
    elif selected_tab == "📚 Business Glossary":
        render_business_glossary()
    elif selected_tab == "🔗 Lineage Analysis":
        render_lineage_analysis()
    elif selected_tab == "📊 Column Impact":
        render_column_impact()
    elif selected_tab == "🕘 Version History":
        render_version_history()

def render_data_exploration():
    """Render the main data exploration interface"""
    dataset_tabs = list(st.session_state.datasets.keys())
    
    if not dataset_tabs:
        st.info("No datasets uploaded yet.")
        return
    
    tabs = st.tabs(dataset_tabs)
    
    for i, (dataset_name, df) in enumerate(st.session_state.datasets.items()):
        with tabs[i]:
            st.header(f"📋 {dataset_name}")
            
            # Check if metadata exists for this dataset
            if dataset_name not in st.session_state.metadata:
                st.error(f"Metadata missing for {dataset_name}. Please re-upload the file.")
                if st.button(f"Regenerate Metadata for {dataset_name}", key=f"regen_{dataset_name}"):
                    # Regenerate metadata for this dataset
                    file_metadata = {}
                    for column in df.columns:
                        col_meta = MetadataExtractor.get_column_metadata(df, column)
                        col_meta['auto_category'] = MetadataExtractor.auto_detect_category(column, col_meta)
                        file_metadata[column] = col_meta
                    
                    st.session_state.metadata[dataset_name] = file_metadata
                    st.success(f"Metadata regenerated for {dataset_name}")
                    st.rerun()
                continue
            
            # Dataset summary with enhanced metrics
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Total Rows", f"{len(df):,}")
            with col2:
                st.metric("Total Columns", len(df.columns))
            with col3:
                missing_cells = df.isnull().sum().sum()
                total_cells = len(df) * len(df.columns)
                missing_pct = (missing_cells / total_cells) * 100
                st.metric("Missing Data", f"{missing_pct:.1f}%")
            with col4:
                memory_usage = df.memory_usage(deep=True).sum() / 1024 / 1024
                st.metric("Memory Usage", f"{memory_usage:.1f} MB")
            with col5:
                # Count outliers across all numeric columns
                total_outliers = 0
                for col in df.select_dtypes(include=[np.number]).columns:
                    outliers = DataProfiler.detect_outliers_iqr(df[col])
                    total_outliers += len(outliers)
                st.metric("Outliers (IQR)", total_outliers)
            
            st.write("---")
            
            # Filters for metadata table
            st.subheader("🔍 Filter & Search")
            filter_col1, filter_col2, filter_col3 = st.columns(3)
            
            with filter_col1:
                data_type_filter = st.selectbox(
                    "Filter by Data Type:",
                    ["All"] + list(df.dtypes.astype(str).unique()),
                    key=f"dtype_filter_{dataset_name}"
                )
            
            with filter_col2:
                category_filter = st.selectbox(
                    "Filter by Category:",
                    ["All", "PII", "Identifier", "Category", "Feature"],
                    key=f"category_filter_{dataset_name}"
                )
            
            with filter_col3:
                missing_threshold = st.slider(
                    "Max Missing %:",
                    0, 100, 100,
                    key=f"missing_filter_{dataset_name}"
                )
            
            # Apply filters
            filtered_columns = []
            for column in df.columns:
                # Skip if metadata doesn't exist for this column
                if column not in st.session_state.metadata[dataset_name]:
                    continue
                    
                meta = st.session_state.metadata[dataset_name][column]
                
                # Data type filter
                if data_type_filter != "All" and meta['data_type'] != data_type_filter:
                    continue
                
                # Category filter
                if category_filter != "All" and meta['auto_category'] != category_filter:
                    continue
                
                # Missing data filter
                if meta['missing_percentage'] > missing_threshold:
                    continue
                
                filtered_columns.append(column)
            
            # Enhanced metadata table
            st.subheader(f"🔍 Column Metadata ({len(filtered_columns)} columns)")
            
            if filtered_columns:
                metadata_rows = []
                for column in filtered_columns:
                    # At this point we know metadata exists (from filtering above)
                    meta = st.session_state.metadata[dataset_name][column]
                    annotations = st.session_state.user_annotations[dataset_name].get(column, {
                        'description': '',
                        'custom_tags': [],
                        'comments': '',
                        'glossary_term': '',
                        'business_owner': ''
                    })
                    
                    # Format tags
                    tags_html = render_tag_pill(meta['auto_category'], meta['auto_category'])
                    for tag in annotations['custom_tags']:
                        tags_html += " " + render_tag_pill(tag, "custom")
                    
                    # Add glossary term if exists
                    if annotations['glossary_term']:
                        tags_html += " " + render_tag_pill(annotations['glossary_term'], "glossary")
                    
                    # Outlier information
                    outlier_info = ""
                    if meta['outliers_iqr'] > 0:
                        outlier_info += f'<span class="outlier-badge">IQR: {meta["outliers_iqr"]}</span> '
                    if meta['outliers_zscore'] > 0:
                        outlier_info += f'<span class="outlier-badge">Z: {meta["outliers_zscore"]}</span>'
                    
                    # Data quality status
                    dq_status = "✅"
                    if dataset_name in st.session_state.data_quality_rules and column in st.session_state.data_quality_rules[dataset_name]:
                        rules = st.session_state.data_quality_rules[dataset_name][column]
                        if rules:
                            rule_results = DataQualityRules.validate_rules(df, column, rules)
                            if any(not result['passed'] for result in rule_results):
                                dq_status = "❌"
                    
                    metadata_rows.append({
                        'Column': column,
                        'Type': meta['data_type'],
                        'Missing %': f"{meta['missing_percentage']:.1f}%",
                        'Unique': f"{meta['unique_count']:,}",
                        'Sample Values': ', '.join(map(str, meta['sample_values'][:3])),
                        'Outliers': outlier_info,
                        'Tags': tags_html,
                        'DQ': dq_status,
                        'Description': (annotations['description'][:50] + "...") if len(annotations['description']) > 50 else annotations['description']
                    })
                
                # Display metadata table
                metadata_df = pd.DataFrame(metadata_rows)
                st.markdown(metadata_df.to_html(escape=False, index=False, classes="metadata-table"), unsafe_allow_html=True)
            else:
                st.warning(f"No columns with metadata found for {dataset_name}.")
                if st.button(f"Regenerate All Metadata for {dataset_name}", key=f"regen_all_{dataset_name}"):
                    # Regenerate metadata for this dataset
                    file_metadata = {}
                    for column in df.columns:
                        col_meta = MetadataExtractor.get_column_metadata(df, column)
                        col_meta['auto_category'] = MetadataExtractor.auto_detect_category(column, col_meta)
                        file_metadata[column] = col_meta
                    
                    st.session_state.metadata[dataset_name] = file_metadata
                    st.success(f"Metadata regenerated for {dataset_name}")
                    st.rerun()
            
            st.write("---")
            
            # Enhanced column details
            st.subheader("📊 Column Deep Dive")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                selected_column = st.selectbox(
                    "Select column for detailed analysis:",
                    df.columns,
                    key=f"select_{dataset_name}"
                )
            
            with col2:
                view_mode = st.radio(
                    "View mode:",
                    ["📊 Visualization", "✏️ Annotations", "✅ Data Quality", "💬 Comments"],
                    horizontal=True,
                    key=f"view_mode_{dataset_name}"
                )
            
            if selected_column:
                # Check if metadata exists for selected column
                if selected_column not in st.session_state.metadata[dataset_name]:
                    st.error(f"Metadata not found for column: {selected_column} in dataset: {dataset_name}")
                    if st.button(f"Regenerate Metadata for {selected_column}", key=f"regen_col_{dataset_name}_{selected_column}"):
                        col_meta = MetadataExtractor.get_column_metadata(df, selected_column)
                        col_meta['auto_category'] = MetadataExtractor.auto_detect_category(selected_column, col_meta)
                        st.session_state.metadata[dataset_name][selected_column] = col_meta
                        st.success(f"Metadata regenerated for {selected_column}")
                        st.rerun()
                    return
                    
                meta = st.session_state.metadata[dataset_name][selected_column]
                
                if view_mode == "📊 Visualization":
                    render_column_visualization(df, selected_column, meta, dataset_name)
                elif view_mode == "✏️ Annotations":
                    render_column_annotations(dataset_name, selected_column)
                elif view_mode == "✅ Data Quality":
                    render_data_quality_rules(df, dataset_name, selected_column)
                elif view_mode == "💬 Comments":
                    render_column_comments(dataset_name, selected_column)
            
            st.write("---")
            
            # Data preview
            st.subheader("👀 Data Preview")
            preview_rows = st.slider("Number of rows to preview:", 5, 50, 10, key=f"preview_{dataset_name}")
            st.dataframe(df.head(preview_rows), use_container_width=True)

def render_column_visualization(df: pd.DataFrame, column: str, meta: Dict, dataset_name: str):
    """Render enhanced column visualization"""
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Main visualization
        fig = DataProfiler.create_column_visualization(df, column)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Column statistics
        st.write("**📈 Statistics**")
        st.write(f"**Data Type:** {meta['data_type']}")
        st.write(f"**Total Values:** {meta['total_rows']:,}")
        st.write(f"**Unique Values:** {meta['unique_count']:,}")
        st.write(f"**Missing Values:** {meta['missing_count']:,} ({meta['missing_percentage']:.1f}%)")
        
        if meta['stats']:
            st.write("**📊 Numeric Stats**")
            for stat, value in meta['stats'].items():
                if not pd.isna(value):
                    st.write(f"**{stat.title()}:** {value:.2f}")
        
        # Outlier information
        if meta['outliers_iqr'] > 0 or meta['outliers_zscore'] > 0:
            st.write("**🚨 Outliers Detected**")
            if meta['outliers_iqr'] > 0:
                st.write(f"IQR Method: {meta['outliers_iqr']} outliers")
                if meta['outlier_indices_iqr']:
                    st.write(f"Sample indices: {meta['outlier_indices_iqr'][:5]}")
            
            if meta['outliers_zscore'] > 0:
                st.write(f"Z-Score Method: {meta['outliers_zscore']} outliers")
                if meta['outlier_indices_zscore']:
                    st.write(f"Sample indices: {meta['outlier_indices_zscore'][:5]}")

def render_column_annotations(dataset_name: str, selected_column: str):
    """Render column annotation interface"""
    # Ensure annotations exist for this column
    if dataset_name not in st.session_state.user_annotations:
        st.session_state.user_annotations[dataset_name] = {}
    
    if selected_column not in st.session_state.user_annotations[dataset_name]:
        st.session_state.user_annotations[dataset_name][selected_column] = {
            'description': '',
            'custom_tags': [],
            'comments': '',
            'glossary_term': '',
            'business_owner': ''
        }
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Description
        description = st.text_area(
            "Description:",
            value=st.session_state.user_annotations[dataset_name][selected_column]['description'],
            key=f"desc_{dataset_name}_{selected_column}",
            help="Describe what this column represents",
            height=100
        )
        
        # Business owner
        business_owner = st.text_input(
            "Business Owner:",
            value=st.session_state.user_annotations[dataset_name][selected_column]['business_owner'],
            key=f"owner_{dataset_name}_{selected_column}",
            help="Who is responsible for this data?"
        )
        
        # Comments
        comments = st.text_area(
            "Additional Notes:",
            value=st.session_state.user_annotations[dataset_name][selected_column]['comments'],
            key=f"comments_{dataset_name}_{selected_column}",
            help="Add any additional notes or observations",
            height=80
        )
    
    with col2:
        # Auto-detected category
        if selected_column in st.session_state.metadata[dataset_name]:
            auto_category = st.session_state.metadata[dataset_name][selected_column]['auto_category']
            st.write(f"**Auto-detected:** {render_tag_pill(auto_category, auto_category)}", unsafe_allow_html=True)
        else:
            st.warning("Metadata not available for this column. Use the regenerate button above.")
        
        # Glossary term mapping
        glossary_terms = list(st.session_state.glossary_terms.keys())
        current_term = st.session_state.user_annotations[dataset_name][selected_column]['glossary_term']
        
        selected_term = st.selectbox(
            "Map to Glossary Term:",
            [""] + glossary_terms,
            index=(glossary_terms.index(current_term) + 1) if current_term in glossary_terms else 0,
            key=f"glossary_{dataset_name}_{selected_column}"
        )
        
        # Custom tags
        new_tag = st.text_input(
            "Add custom tag:",
            key=f"new_tag_{dataset_name}_{selected_column}",
            help="Press Enter to add"
        )
        
        if new_tag and st.button("Add Tag", key=f"add_tag_{dataset_name}_{selected_column}"):
            if new_tag not in st.session_state.user_annotations[dataset_name][selected_column]['custom_tags']:
                st.session_state.user_annotations[dataset_name][selected_column]['custom_tags'].append(new_tag)
                st.rerun()
        
        # Display existing custom tags
        current_tags = st.session_state.user_annotations[dataset_name][selected_column]['custom_tags']
        if current_tags:
            st.write("**Custom Tags:**")
            for tag in current_tags:
                col_tag, col_remove = st.columns([3, 1])
                with col_tag:
                    st.write(render_tag_pill(tag, "custom"), unsafe_allow_html=True)
                with col_remove:
                    if st.button("❌", key=f"remove_{dataset_name}_{selected_column}_{tag}"):
                        current_tags.remove(tag)
                        st.rerun()
    
    # Update annotations
    st.session_state.user_annotations[dataset_name][selected_column]['description'] = description
    st.session_state.user_annotations[dataset_name][selected_column]['business_owner'] = business_owner
    st.session_state.user_annotations[dataset_name][selected_column]['comments'] = comments
    st.session_state.user_annotations[dataset_name][selected_column]['glossary_term'] = selected_term
    
    # Save version button
    if st.button("💾 Save as New Version", key=f"save_version_{dataset_name}_{selected_column}"):
        if dataset_name in st.session_state.metadata and dataset_name in st.session_state.user_annotations:
            snapshot = MetadataVersioning.create_snapshot(
                st.session_state.metadata[dataset_name],
                st.session_state.user_annotations[dataset_name],
                f"Updated {selected_column}"
            )
            if dataset_name not in st.session_state.metadata_versions:
                st.session_state.metadata_versions[dataset_name] = []
            st.session_state.metadata_versions[dataset_name].append(snapshot)
            st.success("New version saved!")
        else:
            st.error("Cannot save version: metadata or annotations missing")

def render_data_quality_rules(df: pd.DataFrame, dataset_name: str, selected_column: str):
    """Render data quality rules interface"""
    if dataset_name not in st.session_state.data_quality_rules:
        st.session_state.data_quality_rules[dataset_name] = {}
    
    if selected_column not in st.session_state.data_quality_rules[dataset_name]:
        st.session_state.data_quality_rules[dataset_name][selected_column] = []
    
    rules = st.session_state.data_quality_rules[dataset_name][selected_column]
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.write("**Add New Rule**")
        
        rule_type = st.selectbox(
            "Rule Type:",
            ["unique", "not_null", "regex", "range"],
            key=f"rule_type_{dataset_name}_{selected_column}"
        )
        
        rule_config = {}
        
        if rule_type == "regex":
            pattern = st.text_input(
                "Regex Pattern:",
                placeholder="e.g., ^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$ for email",
                key=f"regex_pattern_{dataset_name}_{selected_column}"
            )
            rule_config['pattern'] = pattern
        
        elif rule_type == "range":
            if pd.api.types.is_numeric_dtype(df[selected_column]):
                col_min, col_max = st.columns(2)
                with col_min:
                    min_val = st.number_input(
                        "Min Value:",
                        value=None,
                        key=f"min_val_{dataset_name}_{selected_column}"
                    )
                with col_max:
                    max_val = st.number_input(
                        "Max Value:",
                        value=None,
                        key=f"max_val_{dataset_name}_{selected_column}"
                    )
                rule_config['min'] = min_val
                rule_config['max'] = max_val
            else:
                st.warning("Range rules only apply to numeric columns")
        
        if st.button("Add Rule", key=f"add_rule_{dataset_name}_{selected_column}"):
            new_rule = {'type': rule_type, **rule_config}
            rules.append(new_rule)
            st.rerun()
    
    with col2:
        st.write("**Current Rules**")
        
        if rules:
            # Validate all rules
            rule_results = DataQualityRules.validate_rules(df, selected_column, rules)
            
            for i, (rule, result) in enumerate(zip(rules, rule_results)):
                rule_display = f"{rule['type']}"
                if rule['type'] == 'regex':
                    rule_display += f" ({rule.get('pattern', 'No pattern')})"
                elif rule['type'] == 'range':
                    range_parts = []
                    if rule.get('min') is not None:
                        range_parts.append(f"min: {rule['min']}")
                    if rule.get('max') is not None:
                        range_parts.append(f"max: {rule['max']}")
                    rule_display += f" ({', '.join(range_parts)})"
                
                status_class = "rule-passed" if result['passed'] else "rule-violation"
                status_icon = "✅" if result['passed'] else "❌"
                
                st.markdown(f"""
                <div class="{status_class}">
                    <strong>{status_icon} {rule_display}</strong><br>
                    Violations: {result['violation_count']:,} / {result['total_checked']:,}<br>
                    <button onclick="remove_rule({i})">Remove</button>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button("Remove", key=f"remove_rule_{dataset_name}_{selected_column}_{i}"):
                    rules.pop(i)
                    st.rerun()
        else:
            st.info("No data quality rules defined for this column.")

def render_column_comments(dataset_name: str, selected_column: str):
    """Render threaded comments interface"""
    # Initialize comments if they don't exist
    if dataset_name not in st.session_state.comments:
        st.session_state.comments[dataset_name] = {}
    
    if selected_column not in st.session_state.comments[dataset_name]:
        st.session_state.comments[dataset_name][selected_column] = []
        
    comments = st.session_state.comments[dataset_name][selected_column]
    
    # Add new comment
    st.write("**💬 Add Comment**")
    col1, col2 = st.columns([3, 1])
    
    with col1:
        new_comment = st.text_area(
            "Comment:",
            key=f"new_comment_{dataset_name}_{selected_column}",
            height=80
        )
    
    with col2:
        comment_user = st.selectbox(
            "User:",
            SAMPLE_USERS,
            key=f"comment_user_{dataset_name}_{selected_column}"
        )
        
        if st.button("Post Comment", key=f"post_comment_{dataset_name}_{selected_column}"):
            if new_comment.strip():
                comment = {
                    'id': str(uuid.uuid4()),
                    'user': comment_user,
                    'text': new_comment,
                    'timestamp': datetime.now().isoformat(),
                    'replies': []
                }
                comments.append(comment)
                st.rerun()
    
    # Display comments
    st.write("**💬 Comments**")
    
    if comments:
        for comment in sorted(comments, key=lambda x: x['timestamp'], reverse=True):
            timestamp = datetime.fromisoformat(comment['timestamp'])
            time_ago = datetime.now() - timestamp
            
            if time_ago.days > 0:
                time_str = f"{time_ago.days} days ago"
            elif time_ago.seconds > 3600:
                time_str = f"{time_ago.seconds // 3600} hours ago"
            else:
                time_str = f"{time_ago.seconds // 60} minutes ago"
            
            st.markdown(f"""
            <div class="comment-box">
                <strong>{comment['user']}</strong> <small>({time_str})</small><br>
                {comment['text']}
            </div>
            """, unsafe_allow_html=True)
            
            # Reply functionality
            if st.button(f"Reply", key=f"reply_{comment['id']}"):
                st.session_state[f"show_reply_{comment['id']}"] = True
            
            if st.session_state.get(f"show_reply_{comment['id']}", False):
                reply_col1, reply_col2 = st.columns([3, 1])
                with reply_col1:
                    reply_text = st.text_input(
                        "Reply:",
                        key=f"reply_text_{comment['id']}"
                    )
                with reply_col2:
                    reply_user = st.selectbox(
                        "User:",
                        SAMPLE_USERS,
                        key=f"reply_user_{comment['id']}"
                    )
                
                if st.button("Post Reply", key=f"post_reply_{comment['id']}"):
                    if reply_text.strip():
                        reply = {
                            'id': str(uuid.uuid4()),
                            'user': reply_user,
                            'text': reply_text,
                            'timestamp': datetime.now().isoformat()
                        }
                        comment['replies'].append(reply)
                        st.session_state[f"show_reply_{comment['id']}"] = False
                        st.rerun()
            
            # Display replies
            for reply in comment['replies']:
                reply_timestamp = datetime.fromisoformat(reply['timestamp'])
                reply_time_ago = datetime.now() - reply_timestamp
                
                if reply_time_ago.days > 0:
                    reply_time_str = f"{reply_time_ago.days} days ago"
                elif reply_time_ago.seconds > 3600:
                    reply_time_str = f"{reply_time_ago.seconds // 3600} hours ago"
                else:
                    reply_time_str = f"{reply_time_ago.seconds // 60} minutes ago"
                
                st.markdown(f"""
                <div style="margin-left: 2rem;">
                    <div class="comment-box">
                        <strong>{reply['user']}</strong> <small>({reply_time_str})</small><br>
                        {reply['text']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("No comments yet. Be the first to add one!")

def render_business_glossary():
    """Render business glossary management"""
    st.header("📚 Business Glossary")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Add New Term")
        
        term_name = st.text_input("Term Name:")
        term_definition = st.text_area("Definition:")
        term_category = st.selectbox("Category:", ["", "Business", "Technical", "Domain-Specific"])
        parent_term = st.selectbox("Parent Term (optional):", [""] + list(st.session_state.glossary_terms.keys()))
        
        if st.button("Add Term"):
            if term_name and term_definition:
                st.session_state.glossary_terms[term_name] = {
                    'definition': term_definition,
                    'category': term_category,
                    'parent': parent_term if parent_term else None,
                    'created_by': random.choice(SAMPLE_USERS),
                    'created_at': datetime.now().isoformat()
                }
                st.success(f"Added term: {term_name}")
                st.rerun()
    
    with col2:
        st.subheader("Current Glossary Terms")
        
        if st.session_state.glossary_terms:
            # Search within glossary
            search_glossary = st.text_input("Search glossary terms:")
            
            filtered_terms = st.session_state.glossary_terms
            if search_glossary:
                filtered_terms = {
                    k: v for k, v in st.session_state.glossary_terms.items()
                    if search_glossary.lower() in k.lower() or search_glossary.lower() in v['definition'].lower()
                }
            
            for term, details in filtered_terms.items():
                with st.expander(f"📖 {term}"):
                    st.write(f"**Definition:** {details['definition']}")
                    st.write(f"**Category:** {details['category']}")
                    if details['parent']:
                        st.write(f"**Parent Term:** {details['parent']}")
                    st.write(f"**Created by:** {details['created_by']}")
                    
                    # Show where this term is used
                    usage_count = 0
                    for dataset_name, annotations in st.session_state.user_annotations.items():
                        for column, ann in annotations.items():
                            if ann['glossary_term'] == term:
                                usage_count += 1
                    
                    st.write(f"**Usage:** {usage_count} columns")
                    
                    if st.button(f"Delete {term}", key=f"delete_term_{term}"):
                        del st.session_state.glossary_terms[term]
                        st.rerun()
        else:
            st.info("No glossary terms defined yet.")

def render_lineage_analysis():
    """Render enhanced lineage analysis"""
    st.header("🔗 Lineage Analysis")
    
    if len(st.session_state.datasets) < 2:
        st.warning("Upload at least 2 datasets to see lineage analysis.")
        return
    
    matches = LineageAnalyzer.find_column_matches(st.session_state.datasets)
    
    if matches:
        # Interactive graph visualization
        st.subheader("📊 Dataset Relationship Graph")
        fig = LineageAnalyzer.create_lineage_graph(st.session_state.datasets, matches)
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed matches table
        st.subheader("🔍 Detected Relationships")
        
        # Filter options
        col1, col2 = st.columns(2)
        with col1:
            confidence_filter = st.selectbox("Filter by Confidence:", ["All", "High", "Medium", "Low"])
        with col2:
            match_type_filter = st.selectbox("Filter by Match Type:", ["All", "Exact Name Match", "Similar Name"])
        
        # Apply filters
        filtered_matches = matches
        if confidence_filter != "All":
            filtered_matches = [m for m in filtered_matches if m['confidence'] == confidence_filter]
        if match_type_filter != "All":
            filtered_matches = [m for m in filtered_matches if m['match_type'] == match_type_filter]
        
        # Display matches
        for match in filtered_matches:
            confidence_color = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}[match['confidence']]
            
            st.markdown(f"""
            **{match['source_dataset']} ↔ {match['target_dataset']}**
            
            - Column(s): `{match['column_name']}`
            - Match Type: {match['match_type']}
            - Confidence: {confidence_color} {match['confidence']}
            - Similarity Score: {match['similarity_score']:.2f}
            
            ---
            """)
    else:
        st.info("No relationships detected between datasets.")

def render_column_impact():
    """Render column impact analysis"""
    st.header("📊 Column Impact Analysis")
    
    impact_analysis = ColumnImpactAnalyzer.analyze_column_impact(st.session_state.datasets)
    
    # Sort by impact score
    sorted_columns = sorted(impact_analysis.items(), key=lambda x: x[1]['impact_score'], reverse=True)
    
    st.subheader("🎯 High-Impact Columns")
    
    for column, impact in sorted_columns[:10]:  # Top 10
        with st.expander(f"📋 {column} (Impact Score: {impact['impact_score']:.1f})"):
            st.write(f"**Appears in datasets:** {', '.join(impact['datasets'])}")
            st.write(f"**Total appearances:** {impact['total_appearances']}")
            
            if impact['similar_columns']:
                st.write(f"**Similar columns:** {', '.join(impact['similar_columns'])}")
            
            # Simulate impact of removal
            st.write("**🚨 Impact if removed:**")
            affected_datasets = impact['datasets']
            st.write(f"- {len(affected_datasets)} dataset(s) would be affected")
            
            for dataset in affected_datasets:
                # Count dependent columns (simplified simulation)
                dependent_count = len([col for col in st.session_state.datasets[dataset].columns 
                                     if LineageAnalyzer._calculate_similarity(column, col) > 0.3])
                st.write(f"  - {dataset}: {dependent_count} potentially dependent columns")

def render_version_history():
    """Render metadata versioning interface"""
    st.header("🕘 Metadata Version History")
    
    if not any(st.session_state.metadata_versions.values()):
        st.info("No version history available yet.")
        return
    
    # Dataset selection for version history
    dataset_name = st.selectbox("Select dataset:", list(st.session_state.metadata_versions.keys()))
    
    if dataset_name and st.session_state.metadata_versions[dataset_name]:
        versions = st.session_state.metadata_versions[dataset_name]
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📋 Version List")
            
            for i, version in enumerate(reversed(versions)):
                timestamp = datetime.fromisoformat(version['timestamp'])
                
                with st.expander(f"v{len(versions)-i}: {version['version_name']} ({timestamp.strftime('%Y-%m-%d %H:%M')})"):
                    st.write(f"**Version ID:** {version['version_id'][:8]}...")
                    st.write(f"**Timestamp:** {timestamp}")
                    
                    # Show summary of changes
                    if i < len(versions) - 1:  # Not the oldest version
                        prev_version = versions[len(versions)-i-2]
                        changes = MetadataVersioning.compare_versions(prev_version, version)
                        
                        if any(changes.values()):
                            st.write("**Changes:**")
                            for change_type, change_list in changes.items():
                                if change_list:
                                    st.write(f"- {change_type.replace('_', ' ').title()}: {len(change_list)}")
                        else:
                            st.write("**No changes detected**")
                    
                    if st.button(f"Restore this version", key=f"restore_{version['version_id']}"):
                        # Restore annotations (metadata is read-only from CSV)
                        st.session_state.user_annotations[dataset_name] = version['annotations'].copy()
                        st.success("Version restored!")
                        st.rerun()
        
        with col2:
            st.subheader("🔍 Version Comparison")
            
            if len(versions) >= 2:
                version_names = [f"v{i+1}: {v['version_name']}" for i, v in enumerate(reversed(versions))]
                
                version1_idx = st.selectbox("Compare version 1:", range(len(version_names)), 
                                          format_func=lambda x: version_names[x], key="v1")
                version2_idx = st.selectbox("Compare version 2:", range(len(version_names)), 
                                          format_func=lambda x: version_names[x], key="v2")
                
                if version1_idx != version2_idx:
                    v1 = versions[len(versions)-1-version1_idx]
                    v2 = versions[len(versions)-1-version2_idx]
                    
                    changes = MetadataVersioning.compare_versions(v1, v2)
                    
                    st.write("**📊 Comparison Results**")
                    
                    for change_type, change_list in changes.items():
                        if change_list:
                            st.write(f"**{change_type.replace('_', ' ').title()}:**")
                            for change in change_list:
                                st.write(f"- {change}")
                    
                    if not any(changes.values()):
                        st.info("No differences found between selected versions.")
            else:
                st.info("Need at least 2 versions to compare.")
    
    # Export functionality
    st.write("---")
    st.subheader("📤 Export Enhanced Metadata")
    
    export_options = st.multiselect(
        "Select what to export:",
        ["Basic Metadata", "Annotations", "Glossary Terms", "Data Quality Rules", "Comments", "Version History", "Lineage Analysis"],
        default=["Basic Metadata", "Annotations"]
    )
    
    export_format = st.radio("Export format:", ["JSON", "CSV"], horizontal=True)
    
    if st.button("Export Selected Data"):
        export_data = {}
        
        if "Basic Metadata" in export_options:
            export_data['metadata'] = st.session_state.metadata
        
        if "Annotations" in export_options:
            export_data['annotations'] = st.session_state.user_annotations
        
        if "Glossary Terms" in export_options:
            export_data['glossary'] = st.session_state.glossary_terms
        
        if "Data Quality Rules" in export_options:
            export_data['data_quality_rules'] = st.session_state.data_quality_rules
        
        if "Comments" in export_options:
            export_data['comments'] = st.session_state.comments
        
        if "Version History" in export_options:
            export_data['version_history'] = st.session_state.metadata_versions
        
        if "Lineage Analysis" in export_options and len(st.session_state.datasets) > 1:
            export_data['lineage'] = LineageAnalyzer.find_column_matches(st.session_state.datasets)
        
        # Add summary statistics
        export_data['summary'] = {
            'total_datasets': len(st.session_state.datasets),
            'total_columns': sum(len(df.columns) for df in st.session_state.datasets.values()),
            'total_rows': sum(len(df) for df in st.session_state.datasets.values()),
            'export_timestamp': datetime.now().isoformat(),
            'glossary_terms_count': len(st.session_state.glossary_terms),
            'total_comments': sum(len(comments) for dataset_comments in st.session_state.comments.values() 
                                for comments in dataset_comments.values())
        }
        
        if export_format == "JSON":
            json_str = json.dumps(export_data, indent=2, default=str)
            st.download_button(
                label="📥 Download Enhanced Metadata (JSON)",
                data=json_str,
                file_name="enhanced_metadata_export.json",
                mime="application/json"
            )
        else:
            # Create comprehensive CSV export
            csv_rows = []
            
            for dataset_name, dataset_meta in st.session_state.metadata.items():
                for column_name, column_meta in dataset_meta.items():
                    annotations = st.session_state.user_annotations.get(dataset_name, {}).get(column_name, {})
                    rules = st.session_state.data_quality_rules.get(dataset_name, {}).get(column_name, [])
                    comments = st.session_state.comments.get(dataset_name, {}).get(column_name, [])
                    
                    row = {
                        'dataset': dataset_name,
                        'column': column_name,
                        'data_type': column_meta['data_type'],
                        'missing_percentage': column_meta['missing_percentage'],
                        'unique_count': column_meta['unique_count'],
                        'auto_category': column_meta['auto_category'],
                        'outliers_iqr': column_meta['outliers_iqr'],
                        'outliers_zscore': column_meta['outliers_zscore'],
                        'description': annotations.get('description', ''),
                        'business_owner': annotations.get('business_owner', ''),
                        'custom_tags': ', '.join(annotations.get('custom_tags', [])),
                        'glossary_term': annotations.get('glossary_term', ''),
                        'data_quality_rules': len(rules),
                        'comments_count': len(comments),
                        'sample_values': ', '.join(map(str, column_meta['sample_values']))
                    }
                    csv_rows.append(row)
            
            csv_df = pd.DataFrame(csv_rows)
            csv_str = csv_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Enhanced Metadata (CSV)",
                data=csv_str,
                file_name="enhanced_metadata_export.csv",
                mime="text/csv"
            )

    # Footer
    st.markdown("""
    <div class="footer">
        <h3>🔍 Mini Metadata Explorer Pro</h3>
        <p>Built with ❤️ to showcase enterprise-grade metadata collaboration.</p>
        <p><strong>Features:</strong> Advanced Profiling • Interactive Lineage • Data Quality Rules • Collaboration • Versioning</p>
        <p>Inspired by Atlan • Powered by Streamlit • Made for data teams</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
