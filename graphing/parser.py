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

# take the first argument as the filename
import sys
if len(sys.argv) > 1:
    filename = sys.argv[1]
else:
    raise ValueError("Please provide a stats file to parse.")
stats = parse_stats_file(filename)

# create a folder named graphs if it does not exist
import os
import pandas as pd
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
    plt.xlabel('Run Index')
    plt.ylabel('PSNR (dB)')
    plt.legend()
    plt.grid()
    plt.savefig(f'graphs/{filename}.png')
    plt.close()
    
    # also plot an improvement graph based on the first run, show the difference from the first run
    plt.figure(figsize=(10, 6))
    plt.plot([y - y_psnr[0] for y in y_psnr], label='Y-PSNR Improvement', marker='o')
    plt.plot([u - u_psnr[0] for u in u_psnr], label='U-PSNR Improvement', marker='o')
    plt.plot([v - v_psnr[0] for v in v_psnr], label='V-PSNR Improvement', marker='o')
    plt.title(f'{title} Improvement from First Run')
    plt.xlabel('Run Index')
    plt.ylabel('PSNR Improvement (dB)')
    plt.legend()
    plt.grid()
    plt.savefig(f'graphs/{filename}_improvement.png')
    plt.close()
plot_psnr(stats, 'Average PSNR per Run', 'average_psnr')

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
    plt.xlabel('Run Index')
    plt.ylabel('SSIM')
    plt.legend()
    plt.grid()
    plt.savefig(f'graphs/{filename}.png')
    plt.close()
    
    # also plot an improvement graph based on the first run, show the difference from the first run
    plt.figure(figsize=(10, 6))
    plt.plot([y - y_ssim[0] for y in y_ssim], label='Y-SSIM Improvement', marker='o')
    plt.plot([u - u_ssim[0] for u in u_ssim], label='U-SSIM Improvement', marker='o')
    plt.plot([v - v_ssim[0] for v in v_ssim], label='V-SSIM Improvement', marker='o')
    plt.title(f'{title} Improvement from First Run')
    plt.xlabel('Run Index')
    plt.ylabel('SSIM Improvement')
    plt.legend()
    plt.grid()
    plt.savefig(f'graphs/{filename}_improvement.png')
    plt.close() 
plot_ssim(stats, 'Average SSIM per Run', 'average_ssim')

# create a graph for bitrate
def plot_bitrate(data, title, filename):
    bitrates = [frame['frames']['bitrate'] for frame in data]
    
    plt.figure(figsize=(10, 6))
    plt.plot(bitrates, label='Bitrate', marker='o')
    
    plt.title(title)
    plt.xlabel('Run Index')
    plt.ylabel('Bitrate (kbps)')
    plt.legend()
    plt.grid()
    plt.savefig(f'graphs/{filename}.png')
    plt.close()
    
    # also plot an improvement graph based on the first run, show the difference from the first run
    plt.figure(figsize=(10, 6))
    plt.plot([b - bitrates[0] for b in bitrates], label='Bitrate Improvement', marker='o')
    plt.title(f'{title} Improvement from First Run')
    plt.xlabel('Run Index')
    plt.ylabel('Bitrate Improvement (kbps)')
    plt.legend()
    plt.grid()
    plt.savefig(f'graphs/{filename}_improvement.png')       
plot_bitrate(stats, 'Bitrate per Run', 'bitrate')

def plot_reward(reward_path='../Output/monitor.monitor.csv'):
    # check if the reward CSV exists
    import os
    if not os.path.exists(reward_path):
        reward_path = '../logs/monitor.monitor.csv'
        if not os.path.exists(reward_path):
            raise FileNotFoundError(f"Reward CSV file not found at {reward_path}")
    
    # Read the reward CSV, skipping the first line (header with #)
    rewards_df = pd.read_csv(reward_path, comment='#')

    # Plot the 'r' column (reward)
    plt.figure(figsize=(10, 6))
    plt.plot(rewards_df['r'], marker='o', label='Reward')
    plt.title('Reward per Episode')
    plt.xlabel('Episode Index')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid()
    plt.savefig('graphs/reward_per_episode.png')
    plt.close()
plot_reward()
    