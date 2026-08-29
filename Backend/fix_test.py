import re
with open('D:\\Agentic BI analyst\\Backend\\test_quality_score.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix the QualityScore usage
content = content.replace('clean_scorer = QualityScore(clean_df)', 
    '''clean_metadata = profile_metadata(clean_df)
    clean_statistics = profile_statistics(clean_df)

    clean_scorer = QualityScoreProfiler(
        df=clean_df,
        metadata_profile=clean_metadata,
        statistics_profile=clean_statistics
    )''')

content = content.replace('clean_result = clean_scorer.calculate()', 'clean_result = clean_scorer.profile()')
content = content.replace('print(f"✓ Clean dataset score: {clean_score}")', 'print(f"[PASS] Clean dataset score: {clean_score}")')

# Fix the unicode checkmarks
content = content.replace('print("✓ Result is a dictionary.")', 'print("[PASS] Result is a dictionary.")')
content = content.replace('print(f"✓ Quality score found: {score}")', 'print(f"[PASS] Quality score found: {score}")')
content = content.replace('print("✓ Quality score is numeric.")', 'print("[PASS] Quality score is numeric.")')
content = content.replace('print("✓ Quality score is between 0 and 100.")', 'print("[PASS] Quality score is between 0 and 100.")')
content = content.replace('print("✓ Original DataFrame was not modified.")', 'print("[PASS] Original DataFrame was not modified.")')

# Fix the possible_score_keys to include overall_quality_score
content = content.replace('''possible_score_keys = [
        "quality_score",
        "score",
        "overall_score",
        "data_quality_score"
    ]''', '''possible_score_keys = [
        "overall_quality_score",
        "quality_score",
        "score",
        "overall_score",
        "data_quality_score"
    ]''')

with open('D:\\Agentic BI analyst\\Backend\\test_quality_score.py', 'w', encoding='utf-8') as f:
    f.write(content)
print('Fixed')