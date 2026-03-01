# Generator Finder
This is a quick python script for finding good generators for the query expansion trick using only a few key-switching keys. The initial trick can be found in [WhisPIR section 3.2](https://eprint.iacr.org/2024/266.pdf#subsection.3.2). I extended this technique further by allowing multiple "generators".

Note that a similar idea/technique is used in [InspiRING](https://eprint.iacr.org/2025/1352.pdf#section.3) as well.

For the single generator case, the code finds the same generator as WhisPIR does. The two generator case can reduce the number of the required rotation by ~6x.

### How to use
- Just evaluate one pair
```bash
python3 two_gen_bfs.py --n 4096 --w 512 --g 5 --h 3
```

- Also find the best single generator (min `tot_rot`) by brute force
```bash
python3 two_gen_bfs.py --n 4096 --w 512 --search_single
```

- Heuristic search for a strong pair $(g,h)$ (min `tot_rot`) among top-M singles:
```bash
python3 two_gen_bfs.py --n 4096 --w 512 --search_pair --top_m 20
```

### Acknowledgement
Thanks ChatGPT 5.2 for helping me to formalize the ideas and writing the code :)))

