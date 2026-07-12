python plots.py --mode peel
python plots.py --mode release

python plots_crack_tips.py

python release_50_plot.py --time 1.0 --ylim-top 21.6 --ylim-bottom=0.0
python release_50_plot.py --time 0.7 --ylim-top 21.6 --ylim-bottom=0.0
python release_50_plot.py --time 0.3 --ylim-top 21.6 --ylim-bottom=0.0 --show-legend

python release_50_sweep.py --time 1.0 --grid-size 50 --max-iter 2000

python release_50_sweep_time.py --gc1 0.35 --grid-size-gc2 50 --grid-size-time 50 --max-iter 2000

python release_50_propagation.py --jc-top 0.35 --jc-bot 0.01 --time 1.0 --max-iter 4 --ylim-top 5.5
python release_50_propagation.py --jc-top 0.35 --jc-bot 0.1 --time 1.0 --max-iter 4 --ylim-top 5.5
python release_50_propagation.py --jc-top 0.35 --jc-bot 0.2 --time 1.0 --max-iter 4 --ylim-top 5.5


