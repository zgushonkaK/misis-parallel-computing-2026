#!/usr/bin/env python3
"""
Script for building histograms from measurement data (XML format)
Usage: python3 visualize.py [path_to_xml_file]
"""

import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os
import sys

class MeasurementAnalyzer:
    def __init__(self, xml_file):
        self.xml_file = xml_file
        self.data = {}
        self.system_info = {}
        self.load_data()

    def load_data(self):
        try:
            tree = ET.parse(self.xml_file)
            root = tree.getroot()

            system = root.find('system')
            if system is not None:
                self.system_info['cpu_freq'] = float(system.find('cpu_freq_ghz').text)
                self.system_info['array_size'] = int(system.find('array_size').text)
                self.system_info['num_measurements'] = int(system.find('num_measurements').text)

            for method in root.findall('method'):
                method_name = method.get('name')
                measurements = []

                for m in method.findall('measurement'):
                    measurements.append(float(m.text))

                self.data[method_name] = measurements

            print(f"Data loaded successfully from {self.xml_file}")

        except FileNotFoundError:
            print(f"Error: File {self.xml_file} not found!")
            sys.exit(1)
        except ET.ParseError as e:
            print(f"Error parsing XML: {e}")
            sys.exit(1)

    def calculate_statistics(self, values):
        values = np.array(values)

        stats_dict = {
            'min': np.min(values),
            'max': np.max(values),
            'mean': np.mean(values),
            'median': np.median(values),
            'std': np.std(values, ddof=1),
            'var': np.var(values, ddof=1),
            'q1': np.percentile(values, 25),
            'q3': np.percentile(values, 75)
        }

        n = len(values)
        z = 1.96  # for 95% confidence
        stats_dict['ci_lower'] = stats_dict['mean'] - z * stats_dict['std'] / np.sqrt(n)
        stats_dict['ci_upper'] = stats_dict['mean'] + z * stats_dict['std'] / np.sqrt(n)

        return stats_dict

    def print_statistics_report(self, method_name, values):
        stats_dict = self.calculate_statistics(values)

        print(f"\n{'='*60}")
        print(f"{method_name.upper()} STATISTICS".center(60))
        print(f"{'='*60}")
        print(f"Number of measurements: {len(values)}")
        print(f"Minimum time: {stats_dict['min']:.6f} ms")
        print(f"Maximum time: {stats_dict['max']:.6f} ms")
        print(f"Range: {stats_dict['max'] - stats_dict['min']:.6f} ms")
        print(f"\nMean: {stats_dict['mean']:.6f} ms")
        print(f"Median: {stats_dict['median']:.6f} ms")
        print(f"Variance: {stats_dict['var']:.6f}")
        print(f"Standard deviation: {stats_dict['std']:.6f} ms")
        print(f"\nQuartiles:")
        print(f"  Q1 (25%): {stats_dict['q1']:.6f} ms")
        print(f"  Q2 (50%): {stats_dict['median']:.6f} ms")
        print(f"  Q3 (75%): {stats_dict['q3']:.6f} ms")
        print(f"  IQR: {stats_dict['q3'] - stats_dict['q1']:.6f} ms")
        print(f"\n95% Confidence Interval:")
        print(f"  [{stats_dict['ci_lower']:.6f}, {stats_dict['ci_upper']:.6f}] ms")
        print(f"  Width: {stats_dict['ci_upper'] - stats_dict['ci_lower']:.6f} ms")
        print(f"  Result: {stats_dict['mean']:.6f} ± {(stats_dict['ci_upper'] - stats_dict['mean']):.6f} ms")
        print(f"{'='*60}")

    def plot_histogram(self, method_name, values, filename=None):
        plt.figure(figsize=(10, 6))

        n = len(values)

        if n <= 20:
            num_bins = max(8, int(np.sqrt(n)) * 2)
        elif n <= 50:
            num_bins = min(25, n // 2)
        else:
            q75, q25 = np.percentile(values, [75, 25])
            iqr = q75 - q25
            if iqr > 0:
                bin_width = 2 * iqr / (n ** (1/3))
                num_bins = int((max(values) - min(values)) / bin_width)
                num_bins = max(20, min(40, num_bins))
            else:
                num_bins = 20

        print(f"    Using {num_bins} bins")

        plt.hist(values, bins=num_bins, alpha=0.8, color='steelblue',
                edgecolor='black', linewidth=0.5, rwidth=0.95)

        plt.title(method_name, fontsize=14, pad=15)
        plt.xlabel('ms', fontsize=11)
        plt.ylabel('count', fontsize=11)

        plt.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5)

        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)

        plt.tight_layout()

        if filename:
            plt.savefig(filename, dpi=150, bbox_inches='tight', pad_inches=0.1)
            print(f"Saved: {filename}")

        plt.show()

    def analyze_all(self):
        for method_name, values in self.data.items():
            print(f"\n{'#'*60}")
            print(f"Analyzing {method_name} measurements")
            print(f"{'#'*60}")

            self.print_statistics_report(method_name, values)

            self.plot_histogram(
                method_name,
                values,
                filename=f"histogram_{method_name}.png"
            )

def main():
    if len(sys.argv) > 1:
        xml_file = sys.argv[1]
    else:
        xml_file = "measurements.xml"

    if not os.path.exists(xml_file):
        print(f"Error: {xml_file} not found!")
        print("Please run the C++ program first to generate measurements.")
        print("Usage: python3 visualize.py [path_to_xml_file]")
        sys.exit(1)

    analyzer = MeasurementAnalyzer(xml_file)

    print("\n=== SYSTEM INFORMATION ===")
    print(f"CPU Frequency: {analyzer.system_info['cpu_freq']:.2f} GHz")
    print(f"Array size (N): {analyzer.system_info['array_size']}")
    print(f"Measurements per method: {analyzer.system_info['num_measurements']}")

    analyzer.analyze_all()

    print("\nAnalysis complete! Check the generated PNG files for histograms.")

if __name__ == "__main__":
    main()
