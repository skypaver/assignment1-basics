import heapq


class MaxHeap:
    def __init__(self):
        self.heap = []

    def push(self, val):
        # 存入负值，模拟最大堆
        heapq.heappush(self.heap, -val)

    def pop(self):
        # 取出负值并还原
        return -heapq.heappop(self.heap)

    def peek(self):
        # 返回最大值（不弹出）
        if self.heap:
            return -self.heap[0]
        return None

    def size(self):
        return len(self.heap)

    def is_empty(self):
        return len(self.heap) == 0


# 使用示例
if __name__ == "__main__":
    max_heap = MaxHeap()
    max_heap.push(10)
    max_heap.push(30)
    max_heap.push(20)

    print("堆顶元素:", max_heap.peek())  # 输出 30
    print("弹出元素:", max_heap.pop())  # 输出 30
    print("堆顶元素:", max_heap.peek())  # 输出 20
    print("堆大小:", max_heap.size())  # 输出 2
