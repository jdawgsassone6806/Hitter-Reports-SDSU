import os
import sys

import data_utils as du
import pdf_report as pr
from generate_report import OUTPUT_DIR


def main():
    if len(sys.argv) < 2:
        print("Usage: python generate_all.py GAME_FILE")
        sys.exit(1)

    game_file = sys.argv[1]
    df = du.load_pitch_data(game_file)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for name in du.list_batters(df):
        try:
            gr = du.build_game_report(game_file, name)
            safe_name = gr.batter.replace(", ", "_").replace(" ", "_")
            safe_date = gr.date.replace(" ", "_").replace(",", "")
            out_path = os.path.join(OUTPUT_DIR, f"{safe_name}_{safe_date}_Report.pdf")
            pr.build_pdf(gr, out_path)
            print(f"done: {name} -> {out_path}")
        except Exception as e:
            print(f"skipped {name}: {e}")


if __name__ == "__main__":
    main()
