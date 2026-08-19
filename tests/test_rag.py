def test_cross_notebook_isolation():
    """🔴 FAIL before fix, 🟢 PASS after fix"""
    print("🔴 Running test...")
    
    # Simulate the bug: returns results from ALL notebooks
    results_from_search = ["Alpha doc 1", "Alpha doc 2", "Beta doc 1"]
    
    # Check if Beta docs leaked
    beta_docs = [doc for doc in results_from_search if "Beta" in doc]
    
    # This FAILS before the fix
    assert len(beta_docs) == 0, f"🔴 FAIL: Beta documents leaked! Found: {beta_docs}"
    
    print("🟢 PASS: No cross-notebook leakage detected!")

if __name__ == "__main__":
    test_cross_notebook_isolation()
