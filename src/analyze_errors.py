"""
Deep Analysis of Prediction Errors
Analyzes why the model made incorrect predictions and provides insights.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from data_loader import merge_all_data


def analyze_prediction_errors(
    mismatches_path: str = 'outputs/prediction_mismatches.csv',
    data_dir: str = 'data',
    feature_importance_path: str = 'outputs/credit_model_feature_importance.csv'
):
    """
    Analyze prediction errors in detail
    
    Args:
        mismatches_path: Path to mismatches CSV file
        data_dir: Directory containing original data files
        feature_importance_path: Path to feature importance CSV
    """
    print("=" * 80)
    print("XATOLAR TAHLILI / ERROR ANALYSIS")
    print("=" * 80)
    
    # Load mismatches
    try:
        mismatches = pd.read_csv(mismatches_path)
        print(f"\n✓ Xatolar fayli yuklandi: {len(mismatches)} ta xato topildi\n")
    except FileNotFoundError:
        print(f"❌ Xatolar fayli topilmadi: {mismatches_path}")
        sys.exit(1)
    
    # Analyze error types
    false_negatives = mismatches[
        (mismatches['default_actual'] == 1) & 
        (mismatches['default_predicted'] == 0)
    ]
    false_positives = mismatches[
        (mismatches['default_actual'] == 0) & 
        (mismatches['default_predicted'] == 1)
    ]
    
    print("📊 XATO TURLARI / ERROR TYPES:")
    print("-" * 80)
    print(f"1. FALSE NEGATIVE (Xavfli, lekin 'Xavfsiz' deb belgilangan):")
    print(f"   Soni: {len(false_negatives)} ta")
    print(f"   Foizi: {len(false_negatives)/len(mismatches)*100:.1f}%")
    print(f"   ⚠️  JUDA XAVFLI! Bu mijozlarga kredit berib qo'yasiz!\n")
    
    print(f"2. FALSE POSITIVE (Xavfsiz, lekin 'Xavfli' deb belgilangan):")
    print(f"   Soni: {len(false_positives)} ta")
    print(f"   Foizi: {len(false_positives)/len(mismatches)*100:.1f}%")
    print(f"   ⚠️  Yaxshi mijozlarni yo'qotasiz, lekin xavfsizroq.\n")
    
    # Load original data to analyze features
    print("=" * 80)
    print("📈 XUSUSIYATLAR TAHLILI / FEATURE ANALYSIS")
    print("=" * 80)
    
    try:
        # Load all data using the existing data loader
        print("\n📂 Ma'lumotlarni yuklash...")
        data = merge_all_data(data_dir)
        print(f"   Yuklandi: {len(data)} ta yozuv, {len(data.columns)} ta ustun")
        
        # Ensure customer_id exists
        if 'customer_id' not in data.columns:
            if 'customer_ref' in data.columns:
                data = data.rename(columns={'customer_ref': 'customer_id'})
            elif 'cust_id' in data.columns:
                data = data.rename(columns={'cust_id': 'customer_id'})
            else:
                raise ValueError("customer_id ustuni topilmadi!")
        
        # Load feature importance
        try:
            feat_imp = pd.read_csv(feature_importance_path)
            top_features = feat_imp.head(10)['feature'].tolist()
        except:
            top_features = ['credit_score', 'monthly_income', 'loan_amount', 
                          'interest_rate', 'age']
        
        # Analyze false negatives (most dangerous)
        print("\n🔴 FALSE NEGATIVE TAHLILI (Xavfli, lekin model 'Xavfsiz' dedi):")
        print("-" * 80)
        
        fn_data = data[data['customer_id'].isin(false_negatives['customer_id'])]
        all_defaults = data[data['default'] == 1]
        
        print("\n📊 O'rtacha qiymatlar taqqoslash:\n")
        
        comparison_features = [
            'credit_score', 'monthly_income', 'loan_amount', 
            'interest_rate', 'age', 'employment_years'
        ]
        
        available_features = [f for f in comparison_features if f in fn_data.columns]
        
        if available_features:
            print(f"{'Xususiyat':<25} {'O\'tkazib yuborilgan':<20} {'Barcha default':<20} {'Farq':<15}")
            print("-" * 80)
            
            for feature in available_features:
                if fn_data[feature].notna().sum() > 0:
                    fn_mean = fn_data[feature].mean()
                    all_mean = all_defaults[feature].mean()
                    diff = fn_mean - all_mean
                    diff_pct = (diff / all_mean * 100) if all_mean != 0 else 0
                    
                    print(f"{feature:<25} {fn_mean:<20.2f} {all_mean:<20.2f} {diff_pct:>+6.1f}%")
        
        # Analyze false positives
        print("\n\n🟡 FALSE POSITIVE TAHLILI (Xavfsiz, lekin model 'Xavfli' dedi):")
        print("-" * 80)
        
        fp_data = data[data['customer_id'].isin(false_positives['customer_id'])]
        all_good = data[data['default'] == 0]
        
        print("\n📊 O'rtacha qiymatlar taqqoslash:\n")
        
        if available_features:
            print(f"{'Xususiyat':<25} {'Noto\'g\'ri rad etilgan':<20} {'Barcha xavfsiz':<20} {'Farq':<15}")
            print("-" * 80)
            
            for feature in available_features:
                if fp_data[feature].notna().sum() > 0:
                    fp_mean = fp_data[feature].mean()
                    all_mean = all_good[feature].mean()
                    diff = fp_mean - all_mean
                    diff_pct = (diff / all_mean * 100) if all_mean != 0 else 0
                    
                    print(f"{feature:<25} {fp_mean:<20.2f} {all_mean:<20.2f} {diff_pct:>+6.1f}%")
        
        # Recommendations
        print("\n\n" + "=" * 80)
        print("💡 TAVSIYALAR / RECOMMENDATIONS")
        print("=" * 80)
        
        print("\n1. FALSE NEGATIVE'larni kamaytirish uchun:")
        print("   - Threshold'ni yanada pasaytiring (0.2 → 0.15)")
        print("   - Class weight'ni o'zgartiring (default klasiga ko'proq e'tibor)")
        print("   - Yangi xususiyatlar qo'shing (payment history, debt patterns)")
        
        print("\n2. FALSE POSITIVE'larni kamaytirish uchun:")
        print("   - Model parametrlarini fine-tune qiling")
        print("   - Feature engineering'ni yaxshilang")
        print("   - Ensemble metodlarini sinab ko'ring")
        
        print("\n3. Umumiy yaxshilanishlar:")
        print("   - Ko'proq ma'lumotlar to'plang")
        print("   - Cross-validation'dan foydalaning")
        print("   - Hyperparameter tuning qiling (GridSearch)")
        
        # Risk analysis
        print("\n\n" + "=" * 80)
        print("⚠️  RISK TAHLILI / RISK ANALYSIS")
        print("=" * 80)
        
        avg_loan = data['loan_amount'].mean() if 'loan_amount' in data.columns else 50000
        
        fn_risk = len(false_negatives) * avg_loan * 0.7  # 70% yo'qotish
        fp_risk = len(false_positives) * avg_loan * 0.05  # 5% foyda yo'qotish
        
        print(f"\n💸 Moliyaviy ta'sir (O'rtacha kredit: ${avg_loan:,.0f}):")
        print(f"   False Negative riski: ${fn_risk:,.0f}")
        print(f"   False Positive riski: ${fp_risk:,.0f}")
        print(f"   Umumiy risk:          ${fn_risk + fp_risk:,.0f}")
        
        risk_ratio = fn_risk / fp_risk if fp_risk > 0 else float('inf')
        print(f"\n   ⚖️  Risk nisbati: {risk_ratio:.1f}x")
        print(f"       (False Negative {risk_ratio:.1f} marta ko'proq xavfli!)")
        
    except Exception as e:
        print(f"\n❌ Xato: {str(e)}")
        print("Ma'lumotlar tahlil qilinmadi. Iltimos, ma'lumotlar fayllarini tekshiring.")
    
    print("\n" + "=" * 80)
    print("Tahlil yakunlandi!")
    print("=" * 80)


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Xato prognozlarni chuqur tahlil qilish'
    )
    parser.add_argument(
        '--mismatches',
        default='outputs/prediction_mismatches.csv',
        help='Xatolar fayli manzili'
    )
    parser.add_argument(
        '--data-dir',
        default='data',
        help='Ma\'lumotlar katalogi'
    )
    parser.add_argument(
        '--feature-importance',
        default='outputs/credit_model_feature_importance.csv',
        help='Feature importance fayli'
    )
    
    args = parser.parse_args()
    
    analyze_prediction_errors(
        args.mismatches,
        args.data_dir,
        args.feature_importance
    )


if __name__ == '__main__':
    main()
