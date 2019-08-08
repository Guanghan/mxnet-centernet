import heapq
def heapsort(iterable):
    h = []
    for value in iterable:
        heapq.heappush(h, value)
    return [heapq.heappop(h) for i in range(len(h))]

a = [1,3,2,4,5,0, -1]
print(heapsort(a))

