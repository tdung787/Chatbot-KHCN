import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# ===== THAM SỐ VẬT LÝ =====
g = 9.8
k = 0.25
m = 1.0
t = np.linspace(0, 5, 60)
v = (m * g / k) * (1 - np.exp(-k * t / m))
y = (m * g / k) * (t - (m / k) * (1 - np.exp(-k * t / m)))
y = -y  # đảo dấu: rơi xuống dưới

# ===== THIẾT LẬP PLOT =====
fig, ax = plt.subplots(figsize=(4, 6))
ax.set_xlim(-1, 1)
ax.set_ylim(1, np.min(y) * 1.2)
ax.invert_yaxis()
ax.set_xlabel("Vị trí x (m)")
ax.set_ylabel("Chiều cao (m)")
ax.margins(x=0.1, y=0.1)
fig.subplots_adjust(left=0.2, bottom=0.1, top=0.92)

# ===== KHỞI TẠO VẬT THỂ =====
ball, = ax.plot([0], [y[0]], 'ro', markersize=16)

def init():
    ball.set_data([0], [y[0]])
    return (ball,)

def update(frame):
    idx = int(frame)
    idx = max(0, min(idx, len(y) - 1))
    ball.set_data([0], [y[idx]])
    ax.set_title(f"Rơi tự do có lực cản — t = {t[idx]:.2f}s")
    return (ball,)

ani = FuncAnimation(fig, update, frames=range(len(y)), init_func=init, blit=False)
ani.save("fall_with_drag.gif", writer=PillowWriter(fps=12))
plt.close(fig)

print("✅ Đã tạo GIF: fall_with_drag.gif")
