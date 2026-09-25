import heapq


class MaxHeapItem:
    """
    最大堆包装器
    通过反转比较实现 max heap
    """

    __slots__ = ("value",)

    def __init__(self, value):
        self.value = value

    def __lt__(self, other):
        # 注意这里反过来
        return self.value > other.value


class Heap:
    """
    通用 Heap 封装

    mode:
        "min" -> 最小堆
        "max" -> 最大堆
    """

    def __init__(self, mode="min", data=None):
        if mode not in ("min", "max"):
            raise ValueError(
                "mode must be 'min' or 'max'"
            )

        self.mode = mode
        self.heap = []

        if data:
            self.heap = self._wrap(data)
            heapq.heapify(self.heap)


    def _wrap(self, data):
        """
        根据模式包装数据
        """

        if self.mode == "max":
            return [
                MaxHeapItem(x)
                for x in data
            ]

        return list(data)


    def _unwrap(self, item):
        if self.mode == "max":
            return item.value

        return item


    def push(self, item):
        """
        插入元素 O(log n)
        """

        if self.mode == "max":
            item = MaxHeapItem(item)

        heapq.heappush(
            self.heap,
            item
        )


    def pop(self):
        """
        删除堆顶 O(log n)
        """

        item = heapq.heappop(
            self.heap
        )

        return self._unwrap(item)


    def peek(self):
        """
        查看堆顶 O(1)
        """

        return self._unwrap(
            self.heap[0]
        )


    def size(self):
        return len(self.heap)


    def empty(self):
        return len(self.heap) == 0


    def __len__(self):
        return len(self.heap)