import random, math, matplotlib.pyplot as plt
import os

def dist(a,b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def naive_route(start, points):
    return [start] + points

def greedy_route(start, points):
    route = [start]
    unvisited = points.copy()
    current = start
    while unvisited:
        nxt = min(unvisited, key=lambda p: dist(current,p))
        route.append(nxt)
        unvisited.remove(nxt)
        current = nxt
    return route

def route_length(route):
    total = 0.0
    for i in range(len(route)-1):
        total += dist(route[i], route[i+1])
    return total

# generate random pickup points
start = (0,0)
points = [(random.uniform(0,10), random.uniform(0,10)) for _ in range(12)]

r1 = naive_route(start, points)
r2 = greedy_route(start, points)

len1 = route_length(r1)
len2 = route_length(r2)

print("Naive length:", len1, "Greedy length:", len2)

# plot
plt.figure(figsize=(6,6))
xs, ys = zip(*r1); plt.plot(xs, ys, linestyle='--', marker='o', label=f'Naive {len1:.2f}')
xs, ys = zip(*r2); plt.plot(xs, ys, linestyle='-', marker='s', label=f'Greedy {len2:.2f}')
plt.scatter([start[0]],[start[1]], s=100, label='Start', c='red')
plt.legend()
plt.title('Route comparison')

# make sure folder exists
output_folder = 'simulation'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# plt.savefig(os.path.join(output_folder, 'route_comparison.png'))
plt.savefig('route_comparison.png')

print(f"Saved {os.path.join(output_folder, 'route_comparison.png')}")
