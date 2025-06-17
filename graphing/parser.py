import re

def parse_stats_file(filename):
    with open(filename, 'r') as f:
        content = f.read()

    # Find all frame blocks
    frame_blocks = re.findall(
        r'Total Frames\s+Frame Rate\s+Byte Count\s+Bitrate\s+([\d\s\.]+)\s+([\d\.]+) fps\s+([\d]+)\s+([\d\.]+) kbps',
        content)

    # Find all PSNR/SSIM blocks
    psnr_blocks = re.findall(
        r'Average QP\s+Y-PSNR\s+U-PSNR\s+V-PSNR\s+\|\s+Y-PSNR\s+U-PSNR\s+V-PSNR\s+\|\s+Y-SSIM\s+U-SSIM\s+V-SSIM\s+'
        r'([\d\.]+)\s+([\d\.]+) dB\s+([\d\.]+) dB\s+([\d\.]+) dB\s+\|\s+([\d\.]+) dB\s+([\d\.]+) dB\s+([\d\.]+) dB\s+\|\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)',
        content)

    results = []
    count = max(len(frame_blocks), len(psnr_blocks))
    for i in range(count):
        # Frame block
        if i < len(frame_blocks):
            fb = frame_blocks[i]
            total_frames = int(fb[0].strip())
            frame_rate = float(fb[1])
            byte_count = int(fb[2])
            bitrate = float(fb[3])
        else:
            total_frames = frame_rate = byte_count = bitrate = None

        # PSNR/SSIM block
        if i < len(psnr_blocks):
            pb = psnr_blocks[i]
            average_qp = float(pb[0])
            avg_psnr = {
                'Y': float(pb[1]),
                'U': float(pb[2]),
                'V': float(pb[3]),
            }
            overall_psnr = {
                'Y': float(pb[4]),
                'U': float(pb[5]),
                'V': float(pb[6]),
            }
            avg_ssim = {
                'Y': float(pb[7]),
                'U': float(pb[8]),
                'V': float(pb[9]),
            }
        else:
            average_qp = avg_psnr = overall_psnr = avg_ssim = None

        results.append({
            'frames': {
                'total_frames': total_frames,
                'frame_rate': frame_rate,
                'byte_count': byte_count,
                'bitrate': bitrate,
            },
            'psnr_ssim': {
                'average_qp': average_qp,
                'average_psnr': avg_psnr,
                'overall_psnr': overall_psnr,
                'average_ssim': avg_ssim,
            }
        })

    return results

# Example usage:
stats = parse_stats_file('console.log')

# create a folder named graphs if it does not exist
import os
if not os.path.exists('graphs'):
    os.makedirs('graphs')
    
# create a graph for Y-PSNR, U-PSNR, V-PSNR
import matplotlib.pyplot as plt
def plot_psnr(data, title, filename):
    y_psnr = [frame['psnr_ssim']['average_psnr']['Y'] for frame in data]
    u_psnr = [frame['psnr_ssim']['average_psnr']['U'] for frame in data]
    v_psnr = [frame['psnr_ssim']['average_psnr']['V'] for frame in data]
    
    plt.figure(figsize=(10, 6))
    plt.plot(y_psnr, label='Y-PSNR', marker='o')
    plt.plot(u_psnr, label='U-PSNR', marker='o')
    plt.plot(v_psnr, label='V-PSNR', marker='o')
    
    plt.title(title)
    plt.xlabel('Frame Index')
    plt.ylabel('PSNR (dB)')
    plt.legend()
    plt.grid()
    plt.savefig(f'graphs/{filename}.png')
    plt.close()
plot_psnr(stats, 'Average PSNR per Frame', 'average_psnr')

# create a graph for Y-SSIM, U-SSIM, V-SSIM
def plot_ssim(data, title, filename):
    y_ssim = [frame['psnr_ssim']['average_ssim']['Y'] for frame in data]
    u_ssim = [frame['psnr_ssim']['average_ssim']['U'] for frame in data]
    v_ssim = [frame['psnr_ssim']['average_ssim']['V'] for frame in data]
    
    plt.figure(figsize=(10, 6))
    plt.plot(y_ssim, label='Y-SSIM', marker='o')
    plt.plot(u_ssim, label='U-SSIM', marker='o')
    plt.plot(v_ssim, label='V-SSIM', marker='o')
    
    plt.title(title)
    plt.xlabel('Frame Index')
    plt.ylabel('SSIM')
    plt.legend()
    plt.grid()
    plt.savefig(f'graphs/{filename}.png')
    plt.close()
plot_ssim(stats, 'Average SSIM per Frame', 'average_ssim')

# create a graph for bitrate
def plot_bitrate(data, title, filename):
    bitrates = [frame['frames']['bitrate'] for frame in data]
    
    plt.figure(figsize=(10, 6))
    plt.plot(bitrates, label='Bitrate', marker='o')
    
    plt.title(title)
    plt.xlabel('Frame Index')
    plt.ylabel('Bitrate (kbps)')
    plt.legend()
    plt.grid()
    plt.savefig(f'graphs/{filename}.png')
    plt.close()
plot_bitrate(stats, 'Bitrate per Frame', 'bitrate')