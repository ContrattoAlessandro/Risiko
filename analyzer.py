import pickle
from main_neat import analyze_and_print_architecture
import sys

with open('analyze_out_utf8.txt', 'w', encoding='utf-8') as fout:
    sys.stdout = fout
    with open('neat_best.pkl', 'rb') as f:
        data = pickle.load(f)
    analyze_and_print_architecture(data['genome'], data['config'])
