#!/usr/bin/env python3
"""
Generate header banner with user-selected images.
Usage: python3 generate_header.py '[1, 2, 3, ...]'
"""

import json
import sys
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from io import BytesIO
import math

def generate_header(selected_indices):
    # Load metadata
    with open('dns_clip_web_viewer/metadata.json', 'r') as f:
        data = json.load(f)

    coords = np.array(data['coords'])
    fitness = np.array(data['fitness'])

    selected = [{'idx': idx, 'coord': coords[idx], 'fitness': fitness[idx]}
                for idx in selected_indices]

    print(f"Generating header with {len(selected)} selected images...")

    # Layout constants - wide panoramic header
    banner_width = 1600
    banner_height = 500
    plot_size = 280  # Shorter plot height
    img_size = 95  # Thumbnails
    margin = 15

    # Create a wide elliptical scatter plot to fit panoramic banner
    plot_width = int(plot_size * 2.5)  # Much wider plot
    plot_height = plot_size

    fig, ax = plt.subplots(figsize=(plot_width/100, plot_height/100), dpi=150)
    fig.patch.set_facecolor('white')

    fit_norm = (fitness - fitness.min()) / (fitness.max() - fitness.min())
    ax.scatter(coords[:, 0], coords[:, 1], c=fit_norm, cmap='viridis',
               s=14, alpha=0.8, edgecolors='none')

    # Highlight selected points
    for s in selected:
        ax.scatter(s['coord'][0], s['coord'][1], c='#e63946', s=100,
                   edgecolors='white', linewidths=2, zorder=10)

    # Add padding so dots aren't cut off at edges
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect('auto')  # Allow stretching
    ax.axis('off')
    fig.tight_layout(pad=0)

    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', pad_inches=0.03,
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plot_img = Image.open(buf)
    plt.close()

    # Create the banner
    banner = Image.new('RGB', (banner_width, banner_height), (255, 255, 255))
    draw = ImageDraw.Draw(banner)

    # Place the plot in the center
    plot_x = (banner_width - plot_img.width) // 2
    plot_y = (banner_height - plot_img.height) // 2
    banner.paste(plot_img, (plot_x, plot_y))

    # Position images in a circular/elliptical arrangement around the plot
    # Each image is placed in the direction of its point from the center

    n = len(selected)
    center_x = banner_width // 2
    center_y = banner_height // 2

    # Calculate angle for each selected image based on its position relative to center
    for s in selected:
        x, y = s['coord']
        # Angle from center of plot (0.5, 0.5) to point
        angle = math.atan2(y - 0.5, x - 0.5)
        s['angle'] = angle

    # Sort by angle for consistent ordering
    selected = sorted(selected, key=lambda s: s['angle'])

    # Calculate radius for image placement (elliptical to fit banner aspect ratio)
    radius_x = (banner_width - img_size) // 2 - margin - 10
    radius_y = (banner_height - img_size) // 2 - margin - 10

    # Spread out angles to minimize overlaps while maintaining relative order
    n = len(selected)
    if n > 1:
        # Minimum angular separation to avoid overlap (approximate)
        min_angle_sep = 2 * math.pi / max(n, 12)  # At least 30 degrees apart

        # Adjust angles to ensure minimum separation
        angles = [s['angle'] for s in selected]

        # Iteratively push apart angles that are too close
        for iteration in range(50):
            changed = False
            for i in range(n):
                next_i = (i + 1) % n

                # Angular distance (handling wrap-around)
                diff = angles[next_i] - angles[i]
                if next_i == 0:  # Wrap around
                    diff = (angles[next_i] + 2 * math.pi) - angles[i]

                if diff < min_angle_sep:
                    # Push apart
                    push = (min_angle_sep - diff) / 2
                    angles[i] -= push * 0.3
                    angles[next_i] += push * 0.3
                    changed = True

            if not changed:
                break

        # Update selected with adjusted angles
        for i, s in enumerate(selected):
            s['angle'] = angles[i]

    def draw_line(draw, start, end, color=(180, 80, 80), width=5):
        draw.line([start, end], fill=color, width=width)

    # Calculate initial positions
    positions = []
    for i, img_info in enumerate(selected):
        angle = img_info['angle']
        thumb_center_x = center_x + radius_x * math.cos(angle)
        thumb_center_y = center_y - radius_y * math.sin(angle)
        thumb_x = thumb_center_x - img_size // 2
        thumb_y = thumb_center_y - img_size // 2
        thumb_x = max(margin, min(banner_width - img_size - margin, thumb_x))
        thumb_y = max(margin, min(banner_height - img_size - margin, thumb_y))
        positions.append([thumb_x, thumb_y])

    # Push apart overlapping images
    def boxes_overlap(p1, p2, padding=8):
        return not (p1[0] + img_size + padding < p2[0] or
                    p2[0] + img_size + padding < p1[0] or
                    p1[1] + img_size + padding < p2[1] or
                    p2[1] + img_size + padding < p1[1])

    for iteration in range(100):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                if boxes_overlap(positions[i], positions[j]):
                    # Push apart
                    dx = positions[j][0] - positions[i][0]
                    dy = positions[j][1] - positions[i][1]
                    dist = math.sqrt(dx*dx + dy*dy) + 0.1

                    # Normalize and push
                    push = 5
                    positions[i][0] -= (dx / dist) * push
                    positions[i][1] -= (dy / dist) * push
                    positions[j][0] += (dx / dist) * push
                    positions[j][1] += (dy / dist) * push

                    # Keep in bounds
                    positions[i][0] = max(margin, min(banner_width - img_size - margin, positions[i][0]))
                    positions[i][1] = max(margin, min(banner_height - img_size - margin, positions[i][1]))
                    positions[j][0] = max(margin, min(banner_width - img_size - margin, positions[j][0]))
                    positions[j][1] = max(margin, min(banner_height - img_size - margin, positions[j][1]))
                    moved = True
        if not moved:
            break

    # Sort by distance from center - draw closer images first (underneath)
    # so their lines don't overlap on top of other lines
    def dist_from_center(idx):
        thumb_x, thumb_y = positions[idx]
        thumb_cx = thumb_x + img_size // 2
        thumb_cy = thumb_y + img_size // 2
        return math.sqrt((thumb_cx - center_x)**2 + (thumb_cy - center_y)**2)

    draw_order = sorted(range(n), key=dist_from_center)

    # FIRST: Draw all lines (so they appear underneath images)
    for i in draw_order:
        img_info = selected[i]
        thumb_x, thumb_y = positions[i]

        # Calculate point position in plot
        point_x = plot_x + int(img_info['coord'][0] * plot_img.width)
        point_y = plot_y + int((1 - img_info['coord'][1]) * plot_img.height)

        # Draw line from image edge toward the point
        thumb_cx = thumb_x + img_size // 2
        thumb_cy = thumb_y + img_size // 2

        # Line from thumbnail center toward point, starting at edge
        dx = point_x - thumb_cx
        dy = point_y - thumb_cy
        dist = math.sqrt(dx*dx + dy*dy)
        if dist > 0:
            # Start point: edge of thumbnail
            start_x = thumb_cx + (dx / dist) * (img_size // 2 + 4)
            start_y = thumb_cy + (dy / dist) * (img_size // 2 + 4)
            draw_line(draw, (start_x, start_y), (point_x, point_y), color=(180, 80, 80), width=2)

    # SECOND: Draw all images on top of lines
    for i in range(n):
        img_info = selected[i]
        thumb_x, thumb_y = positions[i]

        # Draw white background/border for depth (slightly larger than image)
        border_pad = 4
        draw.rectangle([thumb_x - border_pad, thumb_y - border_pad,
                        thumb_x + img_size + border_pad, thumb_y + img_size + border_pad],
                       fill=(255, 255, 255), outline=(200, 200, 200), width=1)

        # Load and place image
        img_path = f"dns_clip_web_viewer/images/{img_info['idx']}.jpg"
        thumb = Image.open(img_path).resize((img_size, img_size), Image.LANCZOS)
        banner.paste(thumb, (int(thumb_x), int(thumb_y)))

        # Draw thin border around image
        draw.rectangle([thumb_x-1, thumb_y-1, thumb_x+img_size, thumb_y+img_size],
                       outline=(180, 180, 180), width=1)

    # Save
    output_path = 'images/dns_header_banner.png'
    banner.save(output_path, quality=95)
    print(f"Saved to {output_path}")
    print(f"Dimensions: {banner_width}x{banner_height}px")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 generate_header.py '[1, 2, 3, ...]'")
        print("Or:    python3 generate_header.py 1 2 3 ...")
        sys.exit(1)

    # Parse indices from command line
    if sys.argv[1].startswith('['):
        # JSON array format
        indices = json.loads(sys.argv[1])
    else:
        # Space-separated format
        indices = [int(x.strip(',')) for x in sys.argv[1:]]

    generate_header(indices)
