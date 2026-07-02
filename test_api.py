import api
try:
    print("Loading model via api.py...")
    api.load_model()
    print("Done!")
except Exception as e:
    import traceback
    traceback.print_exc()
