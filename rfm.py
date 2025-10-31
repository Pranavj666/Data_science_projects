"""
RFM & Cohort Analysis System
Author: Customer Analytics Team
Description: Comprehensive customer segmentation and retention analysis system
Dataset: Online Retail Dataset (UCI ML Repository)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class RFMCohortAnalyzer:
    """
    A comprehensive class for performing RFM and Cohort Analysis on customer transaction data.
    """
    
    def __init__(self, data_path=None):
        """
        Initialize the analyzer with optional data path.
        
        Parameters:
        -----------
        data_path : str, optional
            Path to the dataset file
        """
        self.df = None
        self.rfm_df = None
        self.cohort_data = None
        self.analysis_date = None
        
        if data_path:
            self.load_data(data_path)
    
    def load_data(self, file_path):
        """
        Load transaction data from CSV or Excel file.
        
        Parameters:
        -----------
        file_path : str
            Path to the data file
        """
        try:
            if file_path.endswith('.csv'):
                self.df = pd.read_csv(file_path, encoding='ISO-8859-1')
            elif file_path.endswith(('.xlsx', '.xls')):
                self.df = pd.read_excel(file_path)
            print(f"✓ Data loaded successfully: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
            return self.df
        except Exception as e:
            print(f"✗ Error loading data: {str(e)}")
            return None
    
    def create_sample_data(self, n_customers=1000, n_transactions=5000):
        """
        Create synthetic transaction data for demonstration.
        
        Parameters:
        -----------
        n_customers : int
            Number of unique customers
        n_transactions : int
            Total number of transactions
        """
        np.random.seed(42)
        
        # Generate date range
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2024, 10, 1)
        date_range = (end_date - start_date).days
        
        # Create transactions
        data = {
            'InvoiceNo': [f'INV{i:06d}' for i in range(1, n_transactions + 1)],
            'CustomerID': np.random.randint(1000, 1000 + n_customers, n_transactions),
            'InvoiceDate': [start_date + timedelta(days=np.random.randint(0, date_range)) 
                           for _ in range(n_transactions)],
            'Quantity': np.random.randint(1, 20, n_transactions),
            'UnitPrice': np.random.uniform(5, 200, n_transactions),
            'Country': np.random.choice(['UK', 'USA', 'Germany', 'France'], n_transactions)
        }
        
        self.df = pd.DataFrame(data)
        self.df['TotalAmount'] = self.df['Quantity'] * self.df['UnitPrice']
        
        print(f"✓ Sample data created: {self.df.shape[0]} transactions for {n_customers} customers")
        return self.df
    
    def preprocess_data(self, date_column='InvoiceDate', customer_column='CustomerID', 
                       amount_columns=None):
        """
        Clean and preprocess the transaction data.
        
        Parameters:
        -----------
        date_column : str
            Name of the date column
        customer_column : str
            Name of the customer ID column
        amount_columns : dict
            Dictionary with 'quantity' and 'price' keys for amount calculation
        """
        print("\n" + "="*60)
        print("DATA PREPROCESSING")
        print("="*60)
        
        df = self.df.copy()
        initial_shape = df.shape
        
        # Convert date column to datetime
        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
        print(f"✓ Converted {date_column} to datetime")
        
        # Remove records with missing CustomerID
        df = df.dropna(subset=[customer_column])
        print(f"✓ Removed {initial_shape[0] - df.shape[0]} rows with missing CustomerID")
        
        # Calculate TotalAmount if not present
        if 'TotalAmount' not in df.columns and amount_columns:
            df['TotalAmount'] = df[amount_columns['quantity']] * df[amount_columns['price']]
            print(f"✓ Calculated TotalAmount from Quantity × UnitPrice")
        
        # Remove negative quantities and prices (returns/cancellations)
        if amount_columns:
            df = df[(df[amount_columns['quantity']] > 0) & (df[amount_columns['price']] > 0)]
        
        # Remove outliers using IQR method
        if 'TotalAmount' in df.columns:
            Q1 = df['TotalAmount'].quantile(0.25)
            Q3 = df['TotalAmount'].quantile(0.75)
            IQR = Q3 - Q1
            df = df[(df['TotalAmount'] >= Q1 - 1.5*IQR) & (df['TotalAmount'] <= Q3 + 1.5*IQR)]
            print(f"✓ Removed outliers using IQR method")
        
        self.df = df
        self.analysis_date = df[date_column].max() + timedelta(days=1)
        
        print(f"\nFinal dataset: {df.shape[0]} rows, {df.shape[1]} columns")
        print(f"Date range: {df[date_column].min().date()} to {df[date_column].max().date()}")
        print(f"Unique customers: {df[customer_column].nunique()}")
        print(f"Analysis reference date: {self.analysis_date.date()}")
        
        return self.df
    
    def calculate_rfm(self, date_column='InvoiceDate', customer_column='CustomerID',
                     amount_column='TotalAmount'):
        """
        Calculate RFM (Recency, Frequency, Monetary) metrics for each customer.
        
        Parameters:
        -----------
        date_column : str
            Name of the date column
        customer_column : str
            Name of the customer ID column
        amount_column : str
            Name of the amount column
        
        Returns:
        --------
        DataFrame with RFM metrics
        """
        print("\n" + "="*60)
        print("RFM CALCULATION")
        print("="*60)
        
        # Calculate RFM metrics
        rfm = self.df.groupby(customer_column).agg({
            date_column: lambda x: (self.analysis_date - x.max()).days,  # Recency
            customer_column: 'count',  # Frequency
            amount_column: 'sum'  # Monetary
        })
        
        rfm.columns = ['Recency', 'Frequency', 'Monetary']
        
        print(f"\nRFM Metrics Summary:")
        print(f"  Recency (days since last purchase):")
        print(f"    Mean: {rfm['Recency'].mean():.1f} days")
        print(f"    Median: {rfm['Recency'].median():.1f} days")
        print(f"\n  Frequency (number of purchases):")
        print(f"    Mean: {rfm['Frequency'].mean():.1f} transactions")
        print(f"    Median: {rfm['Frequency'].median():.1f} transactions")
        print(f"\n  Monetary (total spend):")
        print(f"    Mean: ${rfm['Monetary'].mean():.2f}")
        print(f"    Median: ${rfm['Monetary'].median():.2f}")
        
        self.rfm_df = rfm
        return rfm
    
    def create_rfm_scores(self, r_bins=5, f_bins=5, m_bins=5):
        """
        Create RFM scores by binning metrics into quintiles.
        
        Parameters:
        -----------
        r_bins : int
            Number of bins for Recency (lower is better)
        f_bins : int
            Number of bins for Frequency (higher is better)
        m_bins : int
            Number of bins for Monetary (higher is better)
        """
        # Create scores (1-5, where 5 is best for F and M, but worst for R)
        self.rfm_df['R_Score'] = pd.qcut(self.rfm_df['Recency'], r_bins, 
                                          labels=range(r_bins, 0, -1), duplicates='drop')
        self.rfm_df['F_Score'] = pd.qcut(self.rfm_df['Frequency'], f_bins, 
                                          labels=range(1, f_bins + 1), duplicates='drop')
        self.rfm_df['M_Score'] = pd.qcut(self.rfm_df['Monetary'], m_bins, 
                                          labels=range(1, m_bins + 1), duplicates='drop')
        
        # Convert to integer
        self.rfm_df['R_Score'] = self.rfm_df['R_Score'].astype(int)
        self.rfm_df['F_Score'] = self.rfm_df['F_Score'].astype(int)
        self.rfm_df['M_Score'] = self.rfm_df['M_Score'].astype(int)
        
        # Create RFM Score
        self.rfm_df['RFM_Score'] = (self.rfm_df['R_Score'].astype(str) + 
                                    self.rfm_df['F_Score'].astype(str) + 
                                    self.rfm_df['M_Score'].astype(str))
        
        # Create overall score
        self.rfm_df['RFM_Total'] = (self.rfm_df['R_Score'] + 
                                    self.rfm_df['F_Score'] + 
                                    self.rfm_df['M_Score'])
        
        print(f"\n✓ RFM Scores created (scale: 1-{r_bins})")
        return self.rfm_df
    
    def segment_customers(self):
        """
        Segment customers based on RFM scores into meaningful business categories.
        """
        def assign_segment(row):
            r, f, m = row['R_Score'], row['F_Score'], row['M_Score']
            
            if r >= 4 and f >= 4 and m >= 4:
                return 'Champions'
            elif r >= 3 and f >= 3 and m >= 3:
                return 'Loyal Customers'
            elif r >= 4 and f <= 2 and m <= 2:
                return 'Recent Customers'
            elif r >= 3 and f >= 3:
                return 'Potential Loyalists'
            elif r >= 4 and m >= 4:
                return 'Big Spenders'
            elif r <= 2 and f >= 3:
                return 'At Risk'
            elif r <= 2 and f <= 2 and m >= 3:
                return 'Cant Lose Them'
            elif r <= 2 and f <= 2:
                return 'Lost'
            elif r == 3 and f <= 2:
                return 'About to Sleep'
            else:
                return 'Others'
        
        self.rfm_df['Customer_Segment'] = self.rfm_df.apply(assign_segment, axis=1)
        
        print("\n" + "="*60)
        print("CUSTOMER SEGMENTATION")
        print("="*60)
        
        segment_summary = self.rfm_df.groupby('Customer_Segment').agg({
            'Recency': 'mean',
            'Frequency': 'mean',
            'Monetary': 'mean',
            'Customer_Segment': 'count'
        }).rename(columns={'Customer_Segment': 'Count'})
        
        segment_summary['Percentage'] = (segment_summary['Count'] / 
                                         segment_summary['Count'].sum() * 100)
        
        print("\nSegment Distribution:")
        print(segment_summary.round(2))
        
        return self.rfm_df
    
    def create_cohort_data(self, date_column='InvoiceDate', customer_column='CustomerID'):
        """
        Prepare data for cohort analysis by identifying customer cohorts.
        
        Parameters:
        -----------
        date_column : str
            Name of the date column
        customer_column : str
            Name of the customer ID column
        """
        print("\n" + "="*60)
        print("COHORT ANALYSIS")
        print("="*60)
        
        # Create a copy of the dataframe
        df = self.df.copy()
        
        # Extract year-month for cohort grouping
        df['InvoiceMonth'] = df[date_column].dt.to_period('M')
        
        # Identify first purchase date for each customer (cohort)
        df['CohortMonth'] = df.groupby(customer_column)[date_column].transform('min').dt.to_period('M')
        
        # Calculate cohort index (months since first purchase)
        def get_month_diff(row):
            return (row['InvoiceMonth'] - row['CohortMonth']).n
        
        df['CohortIndex'] = df.apply(get_month_diff, axis=1)
        
        # Create cohort data
        cohort_data = df.groupby(['CohortMonth', 'CohortIndex'])[customer_column].nunique().reset_index()
        cohort_data.rename(columns={customer_column: 'CustomerCount'}, inplace=True)
        
        # Pivot for cohort table
        cohort_table = cohort_data.pivot_table(index='CohortMonth', 
                                               columns='CohortIndex', 
                                               values='CustomerCount')
        
        # Calculate retention rates
        cohort_sizes = cohort_table.iloc[:, 0]
        retention_table = cohort_table.divide(cohort_sizes, axis=0) * 100
        
        self.cohort_data = {
            'cohort_counts': cohort_table,
            'retention_rates': retention_table,
            'cohort_sizes': cohort_sizes
        }
        
        print(f"\n✓ Cohort analysis completed")
        print(f"  Number of cohorts: {len(cohort_sizes)}")
        print(f"  Tracking period: {cohort_table.columns.max()} months")
        print(f"  Average cohort size: {cohort_sizes.mean():.0f} customers")
        
        return self.cohort_data
    
    def visualize_rfm_distribution(self, figsize=(15, 10)):
        """Create visualizations for RFM distribution."""
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle('RFM Metrics Distribution', fontsize=16, fontweight='bold')
        
        # Recency distribution
        axes[0, 0].hist(self.rfm_df['Recency'], bins=50, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Recency Distribution')
        axes[0, 0].set_xlabel('Days Since Last Purchase')
        axes[0, 0].set_ylabel('Number of Customers')
        
        # Frequency distribution
        axes[0, 1].hist(self.rfm_df['Frequency'], bins=50, color='lightcoral', edgecolor='black')
        axes[0, 1].set_title('Frequency Distribution')
        axes[0, 1].set_xlabel('Number of Purchases')
        axes[0, 1].set_ylabel('Number of Customers')
        
        # Monetary distribution
        axes[0, 2].hist(self.rfm_df['Monetary'], bins=50, color='lightgreen', edgecolor='black')
        axes[0, 2].set_title('Monetary Distribution')
        axes[0, 2].set_xlabel('Total Spending ($)')
        axes[0, 2].set_ylabel('Number of Customers')
        
        # RFM Score distribution
        score_dist = self.rfm_df['RFM_Total'].value_counts().sort_index()
        axes[1, 0].bar(score_dist.index, score_dist.values, color='plum', edgecolor='black')
        axes[1, 0].set_title('RFM Total Score Distribution')
        axes[1, 0].set_xlabel('RFM Total Score')
        axes[1, 0].set_ylabel('Number of Customers')
        
        # Segment distribution
        segment_counts = self.rfm_df['Customer_Segment'].value_counts()
        axes[1, 1].barh(segment_counts.index, segment_counts.values, color='coral')
        axes[1, 1].set_title('Customer Segments')
        axes[1, 1].set_xlabel('Number of Customers')
        
        # R vs F scatter
        scatter = axes[1, 2].scatter(self.rfm_df['Recency'], self.rfm_df['Frequency'],
                                    c=self.rfm_df['Monetary'], cmap='viridis', alpha=0.6)
        axes[1, 2].set_title('Recency vs Frequency (Color: Monetary)')
        axes[1, 2].set_xlabel('Recency (Days)')
        axes[1, 2].set_ylabel('Frequency (Purchases)')
        plt.colorbar(scatter, ax=axes[1, 2], label='Monetary ($)')
        
        plt.tight_layout()
        return fig
    
    def visualize_cohort_retention(self, figsize=(14, 8)):
        """Create heatmap visualization for cohort retention."""
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Cohort counts heatmap
        sns.heatmap(self.cohort_data['cohort_counts'], annot=True, fmt='.0f', 
                   cmap='Blues', ax=axes[0], cbar_kws={'label': 'Customer Count'})
        axes[0].set_title('Cohort Analysis - Customer Counts', fontweight='bold')
        axes[0].set_xlabel('Cohort Index (Months)')
        axes[0].set_ylabel('Cohort Month')
        
        # Retention rates heatmap
        sns.heatmap(self.cohort_data['retention_rates'], annot=True, fmt='.1f', 
                   cmap='RdYlGn', ax=axes[1], vmin=0, vmax=100, 
                   cbar_kws={'label': 'Retention Rate (%)'})
        axes[1].set_title('Cohort Analysis - Retention Rates (%)', fontweight='bold')
        axes[1].set_xlabel('Cohort Index (Months)')
        axes[1].set_ylabel('Cohort Month')
        
        plt.tight_layout()
        return fig
    
    def visualize_retention_curves(self, figsize=(12, 6)):
        """Create line plots for retention curves by cohort."""
        fig, ax = plt.subplots(figsize=figsize)
        
        retention = self.cohort_data['retention_rates']
        
        for cohort in retention.index:
            cohort_data = retention.loc[cohort].dropna()
            ax.plot(cohort_data.index, cohort_data.values, marker='o', 
                   label=str(cohort), alpha=0.7)
        
        ax.set_title('Customer Retention Curves by Cohort', fontweight='bold', fontsize=14)
        ax.set_xlabel('Months Since First Purchase', fontsize=12)
        ax.set_ylabel('Retention Rate (%)', fontsize=12)
        ax.legend(title='Cohort', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def generate_insights(self):
        """Generate business insights from RFM and Cohort analysis."""
        print("\n" + "="*60)
        print("KEY BUSINESS INSIGHTS")
        print("="*60)
        
        insights = []
        
        # RFM Insights
        segment_dist = self.rfm_df['Customer_Segment'].value_counts()
        top_segment = segment_dist.index[0]
        top_segment_pct = (segment_dist.iloc[0] / segment_dist.sum() * 100)
        
        insights.append(f"1. Customer Base Composition:")
        insights.append(f"   - Largest segment: {top_segment} ({top_segment_pct:.1f}% of customers)")
        
        champions = self.rfm_df[self.rfm_df['Customer_Segment'] == 'Champions']
        if len(champions) > 0:
            champ_revenue = champions['Monetary'].sum()
            total_revenue = self.rfm_df['Monetary'].sum()
            insights.append(f"   - Champions contribute ${champ_revenue:,.2f} ({champ_revenue/total_revenue*100:.1f}% of total revenue)")
        
        at_risk = self.rfm_df[self.rfm_df['Customer_Segment'].isin(['At Risk', 'Cant Lose Them'])]
        if len(at_risk) > 0:
            insights.append(f"   - {len(at_risk)} high-value customers at risk of churning")
        
        # Cohort Insights
        avg_retention = self.cohort_data['retention_rates'].iloc[:, 1:4].mean().mean()
        insights.append(f"\n2. Retention Performance:")
        insights.append(f"   - Average 3-month retention: {avg_retention:.1f}%")
        
        first_month_retention = self.cohort_data['retention_rates'].iloc[:, 1].mean()
        insights.append(f"   - Month 1 retention: {first_month_retention:.1f}%")
        
        # Recommendations
        insights.append(f"\n3. Strategic Recommendations:")
        insights.append(f"   - Launch reactivation campaigns for 'At Risk' and 'Lost' segments")
        insights.append(f"   - Implement loyalty programs to convert 'Potential Loyalists' to 'Champions'")
        insights.append(f"   - Focus on improving first-month retention with onboarding initiatives")
        insights.append(f"   - Create VIP programs for 'Champions' and 'Loyal Customers'")
        
        for insight in insights:
            print(insight)
        
        return insights
    
    def export_results(self, output_prefix='rfm_cohort_analysis'):
        """Export analysis results to CSV files."""
        self.rfm_df.to_csv(f'{output_prefix}_rfm_results.csv')
        self.cohort_data['cohort_counts'].to_csv(f'{output_prefix}_cohort_counts.csv')
        self.cohort_data['retention_rates'].to_csv(f'{output_prefix}_retention_rates.csv')
        
        print(f"\n✓ Results exported to CSV files with prefix '{output_prefix}'")


# Example usage
if __name__ == "__main__":
    print("RFM & Cohort Analysis System")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = RFMCohortAnalyzer()
    
    # Create sample data (or use analyzer.load_data('your_file.csv'))
    analyzer.create_sample_data(n_customers=1000, n_transactions=5000)
    
    # Preprocess data
    analyzer.preprocess_data(
        date_column='InvoiceDate',
        customer_column='CustomerID',
        amount_columns={'quantity': 'Quantity', 'price': 'UnitPrice'}
    )
    
    # Calculate RFM metrics
    rfm = analyzer.calculate_rfm()
    
    # Create RFM scores and segments
    analyzer.create_rfm_scores()
    analyzer.segment_customers()
    
    # Perform cohort analysis
    analyzer.create_cohort_data()
    
    # Generate visualizations
    fig1 = analyzer.visualize_rfm_distribution()
    plt.savefig('rfm_distribution.png', dpi=300, bbox_inches='tight')
    print("\n✓ RFM distribution chart saved")
    
    fig2 = analyzer.visualize_cohort_retention()
    plt.savefig('cohort_heatmap.png', dpi=300, bbox_inches='tight')
    print("✓ Cohort heatmap saved")
    
    fig3 = analyzer.visualize_retention_curves()
    plt.savefig('retention_curves.png', dpi=300, bbox_inches='tight')
    print("✓ Retention curves saved")
    
    # Generate insights
    analyzer.generate_insights()
    
    # Export results
    analyzer.export_results()
    
    plt.show()
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    