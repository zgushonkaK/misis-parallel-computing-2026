import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np

def parse_results(filename):
    tree = ET.parse(filename)
    root = tree.getroot()

    results = {}
    results['matrix_size'] = int(root.find('matrix_size').text)

    basic = {}
    for alg in root.find('basic_algorithms'):
        basic[alg.get('name')] = float(alg.get('gflops'))
    results['basic'] = basic

    block_results = []
    for bs in root.find('block_size_results'):
        block_results.append((int(bs.get('size')), float(bs.get('gflops'))))
    results['block_sizes'] = sorted(block_results, key=lambda x: x[0])

    unroll_results = []
    for ur in root.find('unrolling_results'):
        unroll_results.append((int(ur.get('m')), float(ur.get('gflops'))))
    results['unrolling'] = sorted(unroll_results, key=lambda x: x[0])

    transpose = root.find('transpose_impact')
    results['transpose_time'] = float(transpose.find('transpose_time').text)
    results['multiply_time_without_transpose'] = float(transpose.find('multiply_time_without_transpose').text)
    results['classic_gflops'] = float(transpose.find('classic_gflops').text)
    results['transpose_gflops'] = float(transpose.find('transpose_gflops').text)

    combined = []
    for comb in root.find('combined_results'):
        combined.append((int(comb.get('block_size')), int(comb.get('unroll_factor')), float(comb.get('gflops'))))
    results['combined'] = combined

    return results

def plot_basic_algorithms(results):
    fig, ax = plt.subplots(figsize=(10, 6))

    names = list(results['basic'].keys())
    values = list(results['basic'].values())

    bars = ax.bar(names, values, color=['gray', 'lightblue', 'lightgreen', 'coral'])
    ax.set_xlabel('Algorithm')
    ax.set_ylabel('Performance (GFLOPS)')
    ax.set_title('Performance Comparison of Basic Matrix Multiplication Algorithms')
    ax.grid(axis='y', alpha=0.3)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.2f}', ha='center', va='bottom')

    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig('basic_algorithms.png', dpi=150)
    plt.show()
    print("Saved: basic_algorithms.png")

def plot_block_size(results):
    fig, ax = plt.subplots(figsize=(12, 6))

    sizes = [x[0] for x in results['block_sizes']]
    gflops = [x[1] for x in results['block_sizes']]

    ax.plot(sizes, gflops, marker='o', linestyle='-', linewidth=2, markersize=8)
    ax.set_xscale('log')
    ax.set_xlabel('Block Size (S)')
    ax.set_ylabel('Performance (GFLOPS)')
    ax.set_title('Performance vs Block Size for Block Matrix Multiplication')
    ax.grid(True, alpha=0.3)

    max_idx = np.argmax(gflops)
    ax.scatter([sizes[max_idx]], [gflops[max_idx]], color='red', s=100, zorder=5)
    ax.annotate(f'Optimal S={sizes[max_idx]}\n{gflops[max_idx]:.2f} GFLOPS',
                xy=(sizes[max_idx], gflops[max_idx]), xytext=(sizes[max_idx]*2, gflops[max_idx]*0.8),
                arrowprops=dict(arrowstyle='->', color='red'))

    plt.tight_layout()
    plt.savefig('block_size_performance.png', dpi=150)
    plt.show()
    print(f"Saved: block_size_performance.png")
    print(f"Optimal block size S* = {sizes[max_idx]} with {gflops[max_idx]:.2f} GFLOPS")

def plot_unrolling(results):
    fig, ax = plt.subplots(figsize=(10, 6))

    unroll = [x[0] for x in results['unrolling']]
    gflops = [x[1] for x in results['unrolling']]

    ax.plot(unroll, gflops, marker='s', linestyle='-', linewidth=2, markersize=8)
    ax.set_xlabel('Unrolling Factor (M)')
    ax.set_ylabel('Performance (GFLOPS)')
    ax.set_title('Performance vs Loop Unrolling for Buffered Multiplication')
    ax.grid(True, alpha=0.3)

    max_idx = np.argmax(gflops)
    ax.scatter([unroll[max_idx]], [gflops[max_idx]], color='red', s=100, zorder=5)
    ax.annotate(f'Optimal M={unroll[max_idx]}\n{gflops[max_idx]:.2f} GFLOPS',
                xy=(unroll[max_idx], gflops[max_idx]), xytext=(unroll[max_idx]+2, gflops[max_idx]*0.85),
                arrowprops=dict(arrowstyle='->', color='red'))

    plt.tight_layout()
    plt.savefig('unrolling_performance.png', dpi=150)
    plt.show()
    print(f"Saved: unrolling_performance.png")
    print(f"Optimal unrolling factor M* = {unroll[max_idx]} with {gflops[max_idx]:.2f} GFLOPS")

def plot_transpose_comparison(results):
    fig, ax = plt.subplots(figsize=(10, 6))

    categories = ['Classic', 'With Transpose\n(total)', 'With Transpose\n(multiplication only)']
    gflops = [results['classic_gflops'], results['transpose_gflops'],
              (2.0 * results['matrix_size']**3) / (results['multiply_time_without_transpose'] * 1e9)]

    bars = ax.bar(categories, gflops, color=['gray', 'lightblue', 'lightgreen'])
    ax.set_ylabel('Performance (GFLOPS)')
    ax.set_title('Performance Comparison: Classic vs Transpose Method')
    ax.grid(axis='y', alpha=0.3)

    for bar, val in zip(bars, gflops):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.2f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('transpose_comparison.png', dpi=150)
    plt.show()
    print("Saved: transpose_comparison.png")

    print("\nTranspose Impact Analysis:")
    print(f"  Classic algorithm: {results['classic_gflops']:.2f} GFLOPS")
    print(f"  Transpose method (total): {results['transpose_gflops']:.2f} GFLOPS")
    print(f"  Transpose method (multiplication only): {(2.0 * results['matrix_size']**3) / (results['multiply_time_without_transpose'] * 1e9):.2f} GFLOPS")
    print(f"  Transpose overhead time: {results['transpose_time']:.2f} seconds")

def plot_combined_heatmap(results):
    fig, ax = plt.subplots(figsize=(12, 8))

    block_sizes = sorted(set([x[0] for x in results['combined']]))
    unroll_factors = sorted(set([x[1] for x in results['combined']]))

    gflops_matrix = np.zeros((len(block_sizes), len(unroll_factors)))
    for S, M, g in results['combined']:
        i = block_sizes.index(S)
        j = unroll_factors.index(M)
        gflops_matrix[i, j] = g

    im = ax.imshow(gflops_matrix, cmap='YlOrRd', aspect='auto', interpolation='nearest')
    ax.set_xticks(np.arange(len(unroll_factors)))
    ax.set_yticks(np.arange(len(block_sizes)))
    ax.set_xticklabels(unroll_factors)
    ax.set_yticklabels(block_sizes)
    ax.set_xlabel('Unrolling Factor (M)')
    ax.set_ylabel('Block Size (S)')
    ax.set_title('Performance Heatmap: Block Size vs Unrolling Factor (GFLOPS)')

    plt.colorbar(im, label='GFLOPS')

    for i in range(len(block_sizes)):
        for j in range(len(unroll_factors)):
            text = ax.text(j, i, f'{gflops_matrix[i, j]:.1f}',
                          ha="center", va="center", color="black", fontsize=9)

    plt.tight_layout()
    plt.savefig('combined_heatmap.png', dpi=150)
    plt.show()
    print("Saved: combined_heatmap.png")

    max_idx = np.unravel_index(np.argmax(gflops_matrix), gflops_matrix.shape)
    print(f"Optimal combination: S* = {block_sizes[max_idx[0]]}, M* = {unroll_factors[max_idx[1]]}")
    print(f"Max performance: {gflops_matrix[max_idx]:.2f} GFLOPS")

def plot_all_algorithms_comparison(results):
    fig, ax = plt.subplots(figsize=(12, 6))

    block_sizes = [x[0] for x in results['block_sizes']]
    block_gflops = [x[1] for x in results['block_sizes']]
    best_block_idx = np.argmax(block_gflops)

    unroll = [x[0] for x in results['unrolling']]
    unroll_gflops = [x[1] for x in results['unrolling']]
    best_unroll_idx = np.argmax(unroll_gflops)

    combined_gflops = {}
    for S, M, g in results['combined']:
        combined_gflops[(S, M)] = g
    best_combined = max(results['combined'], key=lambda x: x[2])

    algorithms = [
        'Classic',
        'Transpose',
        'Buffered',
        f'Block (S={block_sizes[best_block_idx]})',
        f'Buffered (M={unroll[best_unroll_idx]})',
        f'Combined (S={best_combined[0]}, M={best_combined[1]})'
    ]

    performances = [
        results['basic'].get('1. Classic', 0),
        results['basic'].get('2. With transpose', 0),
        results['basic'].get('3. Buffered', 0),
        block_gflops[best_block_idx],
        unroll_gflops[best_unroll_idx],
        best_combined[2]
    ]

    bars = ax.bar(algorithms, performances, color=['gray', 'lightblue', 'lightgreen', 'orange', 'lightcoral', 'gold'])
    ax.set_ylabel('Performance (GFLOPS)')
    ax.set_title('Best Performance Comparison Across All Implementations')
    ax.grid(axis='y', alpha=0.3)

    for bar, val in zip(bars, performances):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.2f}', ha='center', va='bottom')

    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig('best_algorithms_comparison.png', dpi=150)
    plt.show()
    print("Saved: best_algorithms_comparison.png")

    print("\nBest Implementations Summary:")
    print(f"  Classic: {performances[0]:.2f} GFLOPS")
    print(f"  With Transpose: {performances[1]:.2f} GFLOPS")
    print(f"  Buffered: {performances[2]:.2f} GFLOPS")
    print(f"  Block (S={block_sizes[best_block_idx]}): {performances[3]:.2f} GFLOPS")
    print(f"  Buffered with Unrolling (M={unroll[best_unroll_idx]}): {performances[4]:.2f} GFLOPS")
    print(f"  Combined (S={best_combined[0]}, M={best_combined[1]}): {performances[5]:.2f} GFLOPS")
    print(f"\nSpeedup (Combined vs Classic): {performances[5]/performances[0]:.2f}x")

def main():
    try:
        results = parse_results('build/results.xml')
        print("Results loaded successfully")
        print(f"Matrix size: {results['matrix_size']}x{results['matrix_size']}")

        plot_basic_algorithms(results)
        plot_block_size(results)
        plot_unrolling(results)
        plot_transpose_comparison(results)
        plot_combined_heatmap(results)
        plot_all_algorithms_comparison(results)

    except FileNotFoundError:
        print("Error: results.xml not found. Please run the C++ program first to generate results.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
