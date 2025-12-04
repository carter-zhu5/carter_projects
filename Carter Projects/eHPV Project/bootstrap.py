# Ensures the app root is on sys.path when running from /pages in Databricks Apps
import sys, os
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
