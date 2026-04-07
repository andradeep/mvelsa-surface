"""
Visual crop inspection tool.

Shows a scrollable grid of cropped images. Click to mark for deletion (red border).
Press [d] to delete marked images and update labels.csv.
Press [q] or close window to quit without deleting.
Press [n] to go to next page, [p] for previous page.

Usage:
    python inspecionar_crops.py --split train --class BOAT
    python inspecionar_crops.py --split train --data-dir /path/to/coco_cropped

Arguments:
    --split     train / valid / test (default: train)
    --class     class name filter: BOAT, BUOY, LAND, SHIP, SKY (default: all)
    --data-dir  path to coco_cropped directory
    --cols      grid columns (default: 8)
    --rows      grid rows per page (default: 6)
"""

import os
import csv
import argparse
import tkinter as tk
from tkinter import messagebox
from pathlib import Path
from PIL import Image, ImageTk, ImageDraw

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = str(BASE_DIR / "../../../data/coco_cropped")

THUMB_SIZE = 128
BORDER     = 4
PAD        = 2


class CropInspector:
    def __init__(self, images, split_dir, csv_path, cols=8, rows=6):
        self.images      = images  # list of (filename, class_name)
        self.split_dir   = split_dir
        self.csv_path    = csv_path
        self.cols        = cols
        self.rows        = rows
        self.page        = 0
        self.marked      = set()
        self.page_size   = cols * rows

        self.root = tk.Tk()
        self.root.title("Crop Inspector — [d]=delete marked  [n]=next  [p]=prev  [q]=quit")
        self.root.bind('<q>', lambda e: self.root.destroy())
        self.root.bind('<d>', lambda e: self.delete_marked())
        self.root.bind('<n>', lambda e: self.next_page())
        self.root.bind('<p>', lambda e: self.prev_page())

        self.canvas_frame = tk.Frame(self.root)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)

        self.status = tk.Label(self.root, text="", anchor='w', relief=tk.SUNKEN)
        self.status.pack(fill=tk.X)

        self.cell_size = THUMB_SIZE + BORDER * 2 + PAD
        self.canvas_w  = cols * self.cell_size
        self.canvas_h  = rows * self.cell_size

        self.canvas = tk.Canvas(self.canvas_frame, width=self.canvas_w, height=self.canvas_h,
                                 bg='#222')
        self.canvas.pack()
        self.canvas.bind('<Button-1>', self.on_click)

        self.tk_images = []  # keep references
        self.render_page()
        self.root.mainloop()

    def render_page(self):
        self.canvas.delete('all')
        self.tk_images.clear()

        start = self.page * self.page_size
        end   = min(start + self.page_size, len(self.images))
        page_items = self.images[start:end]

        for idx, (filename, class_name) in enumerate(page_items):
            col = idx % self.cols
            row = idx // self.cols
            x0  = col * self.cell_size
            y0  = row * self.cell_size

            img_path = os.path.join(self.split_dir, filename)
            global_idx = start + idx

            try:
                img = Image.open(img_path).resize((THUMB_SIZE, THUMB_SIZE))
            except Exception:
                img = Image.new('RGB', (THUMB_SIZE, THUMB_SIZE), color=(100, 0, 0))

            # Border color
            border_color = '#ff3333' if global_idx in self.marked else '#444'
            bordered = Image.new('RGB',
                                  (THUMB_SIZE + BORDER*2, THUMB_SIZE + BORDER*2),
                                  color=border_color)
            bordered.paste(img, (BORDER, BORDER))

            tk_img = ImageTk.PhotoImage(bordered)
            self.tk_images.append(tk_img)

            self.canvas.create_image(x0, y0, anchor='nw', image=tk_img, tags=f"img_{global_idx}")
            # Class label
            self.canvas.create_text(x0 + BORDER + THUMB_SIZE//2, y0 + THUMB_SIZE + BORDER,
                                     text=f"{class_name[:1]}", fill='#aaa', font=('Arial', 7),
                                     anchor='n')

        total_pages = max(1, (len(self.images) + self.page_size - 1) // self.page_size)
        self.status.config(
            text=f"Page {self.page+1}/{total_pages}  |  "
                 f"Showing {start+1}-{end} of {len(self.images)}  |  "
                 f"Marked for deletion: {len(self.marked)}"
        )

    def on_click(self, event):
        col = event.x // self.cell_size
        row = event.y // self.cell_size
        idx = row * self.cols + col
        global_idx = self.page * self.page_size + idx

        if global_idx >= len(self.images):
            return

        if global_idx in self.marked:
            self.marked.discard(global_idx)
        else:
            self.marked.add(global_idx)

        self.render_page()

    def next_page(self):
        total_pages = (len(self.images) + self.page_size - 1) // self.page_size
        if self.page < total_pages - 1:
            self.page += 1
            self.render_page()

    def prev_page(self):
        if self.page > 0:
            self.page -= 1
            self.render_page()

    def delete_marked(self):
        if not self.marked:
            messagebox.showinfo("Nothing to delete", "No images marked.")
            return

        filenames_to_delete = [self.images[i][0] for i in sorted(self.marked)]
        confirm = messagebox.askyesno(
            "Confirm deletion",
            f"Delete {len(filenames_to_delete)} images?\n" +
            "\n".join(filenames_to_delete[:10]) +
            (f"\n... and {len(filenames_to_delete)-10} more" if len(filenames_to_delete) > 10 else "")
        )
        if not confirm:
            return

        # Delete image files
        deleted = []
        for fn in filenames_to_delete:
            path = os.path.join(self.split_dir, fn)
            if os.path.exists(path):
                os.remove(path)
                deleted.append(fn)

        # Update CSV
        remaining_rows = []
        deleted_set = set(deleted)
        with open(self.csv_path, newline='') as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            for row in reader:
                if row['filename'] not in deleted_set:
                    remaining_rows.append(row)

        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(remaining_rows)

        print(f"Deleted {len(deleted)} images, {len(remaining_rows)} remain in CSV.")

        # Remove from in-memory list
        to_remove = sorted(self.marked, reverse=True)
        for i in to_remove:
            self.images.pop(i)
        self.marked.clear()

        # Adjust page if needed
        total_pages = max(1, (len(self.images) + self.page_size - 1) // self.page_size)
        if self.page >= total_pages:
            self.page = total_pages - 1

        self.render_page()
        messagebox.showinfo("Done", f"Deleted {len(deleted)} images and updated CSV.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split',    default='train',
                        choices=['train', 'valid', 'test'])
    parser.add_argument('--class',    dest='class_name', default=None,
                        choices=['BOAT', 'BUOY', 'LAND', 'SHIP', 'SKY'])
    parser.add_argument('--data-dir', default=DEFAULT_DATA_DIR)
    parser.add_argument('--cols',     type=int, default=8)
    parser.add_argument('--rows',     type=int, default=6)
    args = parser.parse_args()

    split_dir = os.path.join(args.data_dir, args.split)
    csv_path  = os.path.join(split_dir, 'labels.csv')

    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found")
        return

    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    images = [(row['filename'], row['class_name'])
              for row in rows
              if (args.class_name is None or row['class_name'] == args.class_name)
              and os.path.exists(os.path.join(split_dir, row['filename']))]

    if not images:
        print(f"No images found for split={args.split}, class={args.class_name}")
        return

    print(f"Loaded {len(images)} images. Launching inspector...")
    print("  Click image to mark/unmark for deletion")
    print("  [d] delete marked  [n] next page  [p] prev page  [q] quit")

    CropInspector(images, split_dir, csv_path, cols=args.cols, rows=args.rows)


if __name__ == '__main__':
    main()
