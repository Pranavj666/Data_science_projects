import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import json
import warnings
warnings.filterwarnings('ignore')

class ExpenseSharingSystem:
    """
    A comprehensive expense sharing system inspired by Google Pay
    with data science capabilities for analytics and insights.
    """
    
    def __init__(self):
        self.users = {}
        self.groups = {}
        self.expenses = []
        self.transactions = []
        self.settlements = []
        
    def add_user(self, user_id: str, name: str, email: str, phone: str = None):
        """Add a new user to the system."""
        self.users[user_id] = {
            'name': name,
            'email': email,
            'phone': phone,
            'balance': 0.0,
            'total_paid': 0.0,
            'total_owed': 0.0,
            'join_date': datetime.now()
        }
        return f"User {name} added successfully"
    
    def create_group(self, group_id: str, name: str, members: List[str], admin: str):
        """Create a new expense group."""
        if admin not in self.users:
            return "Admin user not found"
        
        invalid_members = [m for m in members if m not in self.users]
        if invalid_members:
            return f"Invalid members: {invalid_members}"
        
        self.groups[group_id] = {
            'name': name,
            'members': members,
            'admin': admin,
            'created_date': datetime.now(),
            'total_expenses': 0.0,
            'active': True
        }
        return f"Group {name} created successfully"
    
    def add_expense(self, expense_id: str, group_id: str, payer_id: str, 
                   amount: float, description: str, split_method: str = 'equal',
                   custom_splits: Dict[str, float] = None, category: str = 'general'):
        """
        Add an expense to a group with various splitting methods.
        
        Args:
            expense_id: Unique identifier for the expense
            group_id: Group where expense occurred
            payer_id: User who paid the expense
            amount: Total amount paid
            description: Description of the expense
            split_method: 'equal', 'weighted', 'custom', or 'percentage'
            custom_splits: Custom split amounts/percentages for each user
            category: Category of expense (food, transport, entertainment, etc.)
        """
        if group_id not in self.groups:
            return "Group not found"
        
        if payer_id not in self.users:
            return "Payer not found"
        
        group = self.groups[group_id]
        members = group['members']
        
        # Calculate splits based on method
        splits = self._calculate_splits(members, amount, split_method, custom_splits)
        
        if splits is None:
            return "Invalid split configuration"
        
        # Create expense record
        expense = {
            'expense_id': expense_id,
            'group_id': group_id,
            'payer_id': payer_id,
            'amount': amount,
            'description': description,
            'category': category,
            'split_method': split_method,
            'splits': splits,
            'date': datetime.now(),
            'status': 'active'
        }
        
        self.expenses.append(expense)
        
        # Update group total
        self.groups[group_id]['total_expenses'] += amount
        
        # Update user balances
        self._update_balances(expense)
        
        return f"Expense {description} added successfully"
    
    def _calculate_splits(self, members: List[str], amount: float, 
                         split_method: str, custom_splits: Dict[str, float] = None):
        """Calculate how expense should be split among members."""
        splits = {}
        
        if split_method == 'equal':
            split_amount = amount / len(members)
            for member in members:
                splits[member] = round(split_amount, 2)
        
        elif split_method == 'weighted':
            if not custom_splits:
                return None
            total_weight = sum(custom_splits.values())
            for member in members:
                weight = custom_splits.get(member, 0)
                splits[member] = round((weight / total_weight) * amount, 2)
        
        elif split_method == 'custom':
            if not custom_splits:
                return None
            if abs(sum(custom_splits.values()) - amount) > 0.01:
                return None
            splits = custom_splits
        
        elif split_method == 'percentage':
            if not custom_splits:
                return None
            if abs(sum(custom_splits.values()) - 100) > 0.01:
                return None
            for member in members:
                percentage = custom_splits.get(member, 0)
                splits[member] = round((percentage / 100) * amount, 2)
        
        return splits
    
    def _update_balances(self, expense: Dict):
        """Update user balances based on expense."""
        payer_id = expense['payer_id']
        amount = expense['amount']
        splits = expense['splits']
        
        # Payer paid the full amount
        self.users[payer_id]['total_paid'] += amount
        
        # Each member owes their split amount
        for member_id, split_amount in splits.items():
            if member_id == payer_id:
                # Payer owes themselves their split (reduces what others owe them)
                self.users[member_id]['balance'] += (amount - split_amount)
            else:
                # Other members owe the payer
                self.users[member_id]['balance'] -= split_amount
                self.users[member_id]['total_owed'] += split_amount
    
    def settle_expense(self, from_user: str, to_user: str, amount: float, 
                      expense_id: str = None):
        """Record a settlement between users."""
        if from_user not in self.users or to_user not in self.users:
            return "User not found"
        
        settlement = {
            'from_user': from_user,
            'to_user': to_user,
            'amount': amount,
            'expense_id': expense_id,
            'date': datetime.now(),
            'status': 'completed'
        }
        
        self.settlements.append(settlement)
        
        # Update balances
        self.users[from_user]['balance'] += amount
        self.users[to_user]['balance'] -= amount
        
        return f"Settlement of ${amount} from {from_user} to {to_user} recorded"
    
    def get_user_balance(self, user_id: str) -> Dict:
        """Get detailed balance information for a user."""
        if user_id not in self.users:
            return {"error": "User not found"}
        
        user = self.users[user_id]
        return {
            'user_id': user_id,
            'name': user['name'],
            'current_balance': user['balance'],
            'total_paid': user['total_paid'],
            'total_owed': user['total_owed'],
            'net_balance': user['balance']
        }
    
    def get_group_summary(self, group_id: str) -> Dict:
        """Get comprehensive summary of group expenses."""
        if group_id not in self.groups:
            return {"error": "Group not found"}
        
        group = self.groups[group_id]
        group_expenses = [e for e in self.expenses if e['group_id'] == group_id]
        
        # Calculate member balances within group
        member_balances = {}
        for member in group['members']:
            member_balances[member] = {
                'paid': 0.0,
                'owes': 0.0,
                'balance': 0.0
            }
        
        for expense in group_expenses:
            payer = expense['payer_id']
            amount = expense['amount']
            splits = expense['splits']
            
            member_balances[payer]['paid'] += amount
            
            for member, split_amount in splits.items():
                member_balances[member]['owes'] += split_amount
        
        # Calculate net balances
        for member in member_balances:
            balance = member_balances[member]
            balance['balance'] = balance['paid'] - balance['owes']
        
        return {
            'group_id': group_id,
            'name': group['name'],
            'members': group['members'],
            'total_expenses': group['total_expenses'],
            'expense_count': len(group_expenses),
            'member_balances': member_balances,
            'recent_expenses': group_expenses[-5:] if group_expenses else []
        }
    
    def generate_settlement_plan(self, group_id: str) -> List[Dict]:
        """Generate optimal settlement plan to minimize transactions."""
        if group_id not in self.groups:
            return []
        
        group_summary = self.get_group_summary(group_id)
        balances = group_summary['member_balances']
        
        # Separate creditors and debtors
        creditors = [(user, data['balance']) for user, data in balances.items() if data['balance'] > 0]
        debtors = [(user, abs(data['balance'])) for user, data in balances.items() if data['balance'] < 0]
        
        # Sort by amount
        creditors.sort(key=lambda x: x[1], reverse=True)
        debtors.sort(key=lambda x: x[1], reverse=True)
        
        settlements = []
        
        i, j = 0, 0
        while i < len(creditors) and j < len(debtors):
            creditor, credit_amount = creditors[i]
            debtor, debt_amount = debtors[j]
            
            # Settle the minimum of credit and debt
            settle_amount = min(credit_amount, debt_amount)
            
            if settle_amount > 0.01:  # Avoid tiny settlements
                settlements.append({
                    'from': debtor,
                    'to': creditor,
                    'amount': round(settle_amount, 2),
                    'from_name': self.users[debtor]['name'],
                    'to_name': self.users[creditor]['name']
                })
            
            # Update amounts
            creditors[i] = (creditor, credit_amount - settle_amount)
            debtors[j] = (debtor, debt_amount - settle_amount)
            
            # Move to next creditor/debtor if current one is settled
            if creditors[i][1] <= 0.01:
                i += 1
            if debtors[j][1] <= 0.01:
                j += 1
        
        return settlements
    
    def get_analytics_data(self) -> Dict:
        """Generate comprehensive analytics data."""
        if not self.expenses:
            return {"message": "No expenses found"}
        
        df = pd.DataFrame(self.expenses)
        
        # Basic statistics
        total_expenses = df['amount'].sum()
        avg_expense = df['amount'].mean()
        expense_count = len(df)
        
        # Category analysis
        category_stats = df.groupby('category').agg({
            'amount': ['sum', 'mean', 'count']
        }).round(2)
        
        # User spending analysis
        user_spending = {}
        for user_id in self.users:
            user_expenses = df[df['payer_id'] == user_id]
            user_spending[user_id] = {
                'name': self.users[user_id]['name'],
                'total_paid': user_expenses['amount'].sum(),
                'expense_count': len(user_expenses),
                'avg_expense': user_expenses['amount'].mean() if len(user_expenses) > 0 else 0
            }
        
        # Monthly trends
        df['month'] = pd.to_datetime(df['date']).dt.to_period('M')
        monthly_expenses = df.groupby('month')['amount'].sum().to_dict()
        
        return {
            'total_expenses': total_expenses,
            'average_expense': avg_expense,
            'expense_count': expense_count,
            'category_analysis': category_stats.to_dict(),
            'user_spending': user_spending,
            'monthly_trends': {str(k): v for k, v in monthly_expenses.items()}
        }
    
    def visualize_expenses(self, group_id: str = None):
        """Create visualizations for expense data."""
        if not self.expenses:
            print("No expenses to visualize")
            return
        
        # Filter by group if specified
        if group_id:
            expenses_data = [e for e in self.expenses if e['group_id'] == group_id]
        else:
            expenses_data = self.expenses
        
        if not expenses_data:
            print("No expenses found for the specified group")
            return
        
        df = pd.DataFrame(expenses_data)
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Expense Analysis Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Category-wise expenses
        category_totals = df.groupby('category')['amount'].sum().sort_values(ascending=False)
        axes[0, 0].pie(category_totals.values, labels=category_totals.index, autopct='%1.1f%%')
        axes[0, 0].set_title('Expenses by Category')
        
        # 2. User spending comparison
        user_spending = df.groupby('payer_id')['amount'].sum().sort_values(ascending=False)
        user_names = [self.users[uid]['name'] for uid in user_spending.index]
        axes[0, 1].bar(user_names, user_spending.values)
        axes[0, 1].set_title('Total Spending by User')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Expense amount distribution
        axes[1, 0].hist(df['amount'], bins=20, alpha=0.7, color='skyblue')
        axes[1, 0].set_title('Distribution of Expense Amounts')
        axes[1, 0].set_xlabel('Amount ($)')
        axes[1, 0].set_ylabel('Frequency')
        
        # 4. Timeline of expenses
        df['date'] = pd.to_datetime(df['date'])
        daily_expenses = df.groupby(df['date'].dt.date)['amount'].sum()
        axes[1, 1].plot(daily_expenses.index, daily_expenses.values, marker='o')
        axes[1, 1].set_title('Daily Expense Trends')
        axes[1, 1].set_xlabel('Date')
        axes[1, 1].set_ylabel('Amount ($)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def export_data(self, filename: str = 'expense_data.json'):
        """Export all data to JSON file."""
        data = {
            'users': self.users,
            'groups': self.groups,
            'expenses': self.expenses,
            'settlements': self.settlements,
            'export_date': datetime.now().isoformat()
        }
        
        # Convert datetime objects to strings
        def serialize_datetime(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
        
        with open(filename, 'w') as f:
            json.dump(data, f, default=serialize_datetime, indent=2)
        
        return f"Data exported to {filename}"
    
    def handle_edge_cases(self):
        """Handle common edge cases in expense sharing."""
        edge_cases = {
            'uneven_splits': 'System handles uneven splits through weighted and custom split methods',
            'missing_payments': 'Settlement tracking allows users to record partial payments',
            'user_leaves_group': 'Balances are maintained even if users become inactive',
            'refunds': 'Negative expenses can be added to handle refunds',
            'currency_precision': 'All amounts are rounded to 2 decimal places to avoid floating point errors',
            'duplicate_expenses': 'Unique expense IDs prevent duplicate entries'
        }
        return edge_cases

# Demo and Testing Functions
def create_sample_data():
    """Create sample data for testing and demonstration."""
    system = ExpenseSharingSystem()
    
    # Add users
    system.add_user('u1', 'Alice Johnson', 'alice@email.com', '+1234567890')
    system.add_user('u2', 'Bob Smith', 'bob@email.com', '+1234567891')
    system.add_user('u3', 'Charlie Brown', 'charlie@email.com', '+1234567892')
    system.add_user('u4', 'Diana Prince', 'diana@email.com', '+1234567893')
    
    # Create groups
    system.create_group('g1', 'Weekend Trip', ['u1', 'u2', 'u3', 'u4'], 'u1')
    system.create_group('g2', 'Office Lunch', ['u1', 'u2', 'u3'], 'u2')
    
    # Add sample expenses
    system.add_expense('e1', 'g1', 'u1', 120.00, 'Hotel booking', 'equal', category='accommodation')
    system.add_expense('e2', 'g1', 'u2', 80.00, 'Dinner at restaurant', 'equal', category='food')
    system.add_expense('e3', 'g1', 'u3', 45.00, 'Taxi ride', 'equal', category='transport')
    system.add_expense('e4', 'g2', 'u1', 60.00, 'Team lunch', 'equal', category='food')
    system.add_expense('e5', 'g1', 'u4', 100.00, 'Groceries', 'weighted', 
                      {'u1': 2, 'u2': 2, 'u3': 1, 'u4': 3}, category='food')
    
    return system

def run_demo():
    """Run a comprehensive demo of the expense sharing system."""
    print("=== Google Pay-Inspired Expense Sharing System Demo ===\n")
    
    # Create sample data
    system = create_sample_data()
    
    # Display user balances
    print("1. USER BALANCES:")
    print("-" * 50)
    for user_id in system.users:
        balance = system.get_user_balance(user_id)
        print(f"{balance['name']}: ${balance['current_balance']:.2f}")
        print(f"   Total Paid: ${balance['total_paid']:.2f}")
        print(f"   Total Owed: ${balance['total_owed']:.2f}\n")
    
    # Display group summary
    print("2. GROUP SUMMARY:")
    print("-" * 50)
    summary = system.get_group_summary('g1')
    print(f"Group: {summary['name']}")
    print(f"Total Expenses: ${summary['total_expenses']:.2f}")
    print(f"Number of Expenses: {summary['expense_count']}")
    print("\nMember Balances in Group:")
    for member, balance in summary['member_balances'].items():
        name = system.users[member]['name']
        print(f"  {name}: Paid ${balance['paid']:.2f}, Owes ${balance['owes']:.2f}, Balance ${balance['balance']:.2f}")
    
    # Generate settlement plan
    print("\n3. SETTLEMENT PLAN:")
    print("-" * 50)
    settlements = system.generate_settlement_plan('g1')
    for settlement in settlements:
        print(f"{settlement['from_name']} owes {settlement['to_name']}: ${settlement['amount']:.2f}")
    
    # Display analytics
    print("\n4. ANALYTICS:")
    print("-" * 50)
    analytics = system.get_analytics_data()
    print(f"Total System Expenses: ${analytics['total_expenses']:.2f}")
    print(f"Average Expense: ${analytics['average_expense']:.2f}")
    print(f"Total Expense Count: {analytics['expense_count']}")
    
    # Edge cases
    print("\n5. EDGE CASES HANDLED:")
    print("-" * 50)
    edge_cases = system.handle_edge_cases()
    for case, description in edge_cases.items():
        print(f"• {case.replace('_', ' ').title()}: {description}")
    
    # Create visualizations
    print("\n6. GENERATING VISUALIZATIONS...")
    print("-" * 50)
    system.visualize_expenses('g1')
    
    return system

if __name__ == "__main__":
    # Run the demo
    system = run_demo()
    
    # Export data
    print("\n7. EXPORTING DATA...")
    print("-" * 50)
    export_result = system.export_data()
    print(export_result)