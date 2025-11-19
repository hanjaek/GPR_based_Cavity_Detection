import matplotlib.pyplot as plt

methods = ["Manual\n(Human)", "Light U-Net\n(Model)"]
times_hours = [51.9, 12.4 / 3600]   # 51.9h vs 0.0034h
colors = ["red", "#3b82f6"]

plt.figure(figsize=(7,5))
plt.bar(methods, times_hours, color=colors)
plt.yscale("log")   # ★ 로그 스케일

plt.ylabel("Processing Time (hours, log scale)")
plt.title("Total Time to Process 1,558 GPR Images")

for m, t in zip(methods, times_hours):
    plt.text(m, t, f"{t:.4f} h", ha="center", va="bottom")

plt.tight_layout()
plt.show()
