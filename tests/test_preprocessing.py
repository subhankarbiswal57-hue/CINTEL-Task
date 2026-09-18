"""
Unit tests for data preprocessing and feature engineering routines.
"""

import pandas as pd
import pytest
from src.train_models import extract_title, engineer_features, build_preprocessor


def test_extract_title():
    assert extract_title("Braund, Mr. Owen Harris") == "Mr"
    assert extract_title("Cumings, Mrs. John Bradley (Florence Briggs Thayer)") == "Mrs"
    assert extract_title("Heikkinen, Miss. Laina") == "Miss"
    assert extract_title("Futrelle, Mlle. Lily") == "Miss"
    assert extract_title("Reynaldo, Ms. Encarnacion") == "Miss"
    assert extract_title("Duff Gordon, Lady. (Lucille Christiana)") == "Rare"


def test_engineer_features():
    df = pd.DataFrame({
        'Name': ['Braund, Mr. Owen Harris', 'Heikkinen, Miss. Laina', 'Johnson, Mrs. Oscar W (Elisabeth Vilhelmina)'],
        'SibSp': [1, 0, 0],
        'Parch': [0, 0, 2]
    })
    df_engineered = engineer_features(df)
    
    assert 'Title' in df_engineered.columns
    assert 'FamilySize' in df_engineered.columns
    assert 'IsAlone' in df_engineered.columns
    
    # Mr: SibSp=1, Parch=0 -> FamilySize=2, IsAlone=0
    assert df_engineered.loc[0, 'FamilySize'] == 2
    assert df_engineered.loc[0, 'IsAlone'] == 0
    
    # Miss: SibSp=0, Parch=0 -> FamilySize=1, IsAlone=1
    assert df_engineered.loc[1, 'FamilySize'] == 1
    assert df_engineered.loc[1, 'IsAlone'] == 1


def test_build_preprocessor():
    preprocessor, num_cols, cat_cols = build_preprocessor()
    assert 'Age' in num_cols
    assert 'Sex' in cat_cols
    assert preprocessor is not None
