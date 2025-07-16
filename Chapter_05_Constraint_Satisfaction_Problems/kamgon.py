"""
Chapter_02_Search_Strategies, Chapter_05_Constraint_Satisfaction_Problems
โมดูลนี้ใช้อัลกอริทึมการค้นหาแบบต่างๆ (ทั้งแบบที่ "ไม่รู้ข้อมูล" และแบบที่ "รู้ข้อมูล")
เพื่อช่วยแก้ปัญหาที่กำหนดอยู่ใน "พื้นที่สถานะ" (state space)
โดยมีคลาสพื้นฐาน (abstract class) สำหรับ:
- การกำหนดปัญหา (Problem)
- โหนด (Node) ในแผนผังการค้นหา (search tree)
- ประเภทต่างๆ ของ "คิว" (waitlist) เช่น Stack, Queue, PriorityQueue

ส่วนประกอบหลักๆ ของโมดูลนี้คือ:
- Problem: คลาสหลักที่ใช้กำหนด "ปัญหาการค้นหา" คุณต้องกำหนดเมธอดสำหรับ "การกระทำ" (actions), "ผลลัพธ์" (result), "การทดสอบเป้าหมาย" (goal_test), "ต้นทุนเส้นทาง" (path_cost), และ "ฮิวริสติก" (heuristic)
- Node: แสดงถึง "โหนด" ในแผนผังการค้นหา โดยจะเก็บข้อมูล "สถานะ" (state) ปัจจุบัน, "โหนดแม่" (parent), "การกระทำ" (action) ที่ทำให้มาถึงสถานะนี้, "ต้นทุนเส้นทาง" (path_cost) จากจุดเริ่มต้น และ "ความลึก" (depth)
- Waitlist: คลาสพื้นฐานสำหรับกลยุทธ์การจัดการคิวต่างๆ ที่ใช้ในอัลกอริทึมการค้นหา
- Stack (LIFO - Last-In, First-Out): ใช้สำหรับการค้นหาแบบ "เจาะลึก" (Depth-First Search หรือ DFS)
- Queue (FIFO - First-In, First-Out): ใช้สำหรับการค้นหาแบบ "เจาะกว้าง" (Breadth-First Search หรือ BFS)
- PriorityQueue: ใช้สำหรับอัลกอริทึมการค้นหาแบบ "รู้ข้อมูล" (informed search) เช่น Best-First Search, Uniform Cost Search, Greedy Best-First Search และ A* Search

อัลกอริทึมการค้นหาที่ถูกนำมาใช้:
- uninform_search: เป็นฟังก์ชันการค้นหาแบบ "ไม่รู้ข้อมูล" ทั่วไป สามารถทำได้ทั้ง BFS (ถ้าใช้ Queue เป็น waitlist) หรือ DFS (ถ้าใช้ Stack)
- best_first_search: เป็นฟังก์ชันการค้นหาแบบ "รู้ข้อมูล" ทั่วไป ซึ่งใช้ PriorityQueue และฟังก์ชันประเมินผล "f" ที่กำหนดเอง
- greedy_best_first_search: ใช้วิธี Best-First Search แบบ "โลภ" (greedy) โดยใช้แค่ฟังก์ชัน "ฮิวริสติก" h(n) เป็นฟังก์ชันประเมินผล
- uniform_cost_search: ใช้วิธี Uniform Cost Search (UCS) โดยใช้แค่ "ต้นทุนเส้นทาง" g(n) เป็นฟังก์ชันประเมินผล
- astar_search: ใช้วิธี A* Search โดยใช้ฟังก์ชันประเมินผล f(n) = g(n) + h(n)

โมดูลนี้ยังมีตัวอย่างการใช้งาน ที่แสดงวิธีสร้างปัญหา "กราฟ" (graph) แบบง่ายๆ และแก้ปัญหาด้วย A* Search
"""

import heapq
from collections import deque
from typing import TypeVar, List, Dict, Any, Optional, Generic, Callable, Set, Deque, Tuple

# กำหนดประเภททั่วไปสำหรับสถานะ (state)
S = TypeVar('S')
# กำหนดประเภททั่วไปสำหรับการกระทำ (action)
A = TypeVar('A')

class Problem(Generic[S, A]):
    """
    คลาสหลัก (Abstract class) สำหรับการกำหนดปัญหาการค้นหาอย่างเป็นทางการ
    คุณต้องสร้างคลาสย่อยและใส่โค้ดการทำงานในเมธอดต่างๆ เพื่อให้เป็นปัญหาเฉพาะของคุณ
    """
    def actions(self, state: S) -> List[A]:
        """
        คืนค่า "การกระทำ" (actions) ที่สามารถทำได้ใน "สถานะ" (state) ปัจจุบันที่กำหนด
        :param state: สถานะปัจจุบัน
        :return: รายการของการกระทำที่สามารถทำได้จากสถานะปัจจุบัน
        """
        raise NotImplementedError

    def result(self, state: S, action: A) -> S:
        """
        คืนค่า "สถานะ" (state) ที่เป็นผลจากการทำ "การกระทำ" (action) ใน "สถานะ" (state) ปัจจุบันที่กำหนด
        :param state: สถานะปัจจุบัน
        :param action: การกระทำที่จะทำ
        :return: สถานะใหม่หลังจากทำตามการกระทำ
        """
        raise NotImplementedError

    def goal_test(self, state: S, goal: S) -> bool:
        """
        คืนค่า True ถ้า "สถานะ" (state) เป็น "สถานะเป้าหมาย" (goal state)
        :param state: สถานะปัจจุบันที่จะตรวจสอบ
        :param goal: สถานะเป้าหมายที่ต้องการเปรียบเทียบ
        :return: True ถ้าสถานะปัจจุบันคือสถานะเป้าหมาย, ไม่อย่างนั้นคืนค่า False
        """
        return state == goal

    def path_cost(self, current_cost: float, state1: S, action: A, state2: S) -> float:
        """
        คืนค่า "ต้นทุนของเส้นทาง" (path cost) ของเส้นทางที่ไปถึง state2
        จาก state1 โดยทำ "การกระทำ" (action) และมีต้นทุน c ในการไปถึง state1
        โดยค่าเริ่มต้น ต้นทุนจะเท่ากับ 1 สำหรับแต่ละขั้นตอน
        :param current_cost: ต้นทุนทั้งหมดที่ใช้ไปเพื่อไปถึง state1
        :param state1: สถานะก่อนหน้า
        :param action: การกระทำที่ทำเพื่อย้ายจาก state1 ไปยัง state2
        :param state2: สถานะปัจจุบันที่ไปถึง
        :return: ต้นทุนเส้นทางรวมในการไปถึง state2
        """
        return current_cost + 1

    def heuristic(self, state: S, goal: S) -> float:
        """
        คืนค่า "ฮิวริสติก" (heuristic value) ซึ่งเป็นค่าประมาณของต้นทุนที่เหลือไปถึงเป้าหมาย สำหรับ "สถานะ" (state) ที่กำหนด
        ฟังก์ชันนี้ควรถูกเขียนทับ (override) โดยคลาสย่อย สำหรับอัลกอริทึมการค้นหาแบบ "รู้ข้อมูล" (informed search)
        ค่าฮิวริสติกเริ่มต้นเป็น 0 ทำให้ฟังก์ชันนี้เหมาะสำหรับการค้นหาแบบ "ไม่รู้ข้อมูล" (uninformed search)
        เมื่อใช้อัลกอริทึมที่ต้องการฮิวริสติก (เช่น A* จะทำงานเหมือน UCS ถ้าไม่ระบุฮิวริสติก)
        :param state: สถานะปัจจุบัน
        :param goal: สถานะเป้าหมาย
        :return: ค่าประมาณต้นทุนจากสถานะปัจจุบันไปยังสถานะเป้าหมาย
        """
        return 0.0

    def view_state(self, state: S = None):
        """
        เมธอดสำหรับแสดงภาพหรือพิมพ์สถานะ
        เมธอดนี้เป็นทางเลือก (optional) และสามารถนำไปใช้ในคลาสย่อยเพื่อการดีบัก (debugging) หรือการแสดงผล
        :param state: สถานะที่จะแสดงภาพ ถ้าเป็น None อาจจะแสดงภาพสถานะปัจจุบันของปัญหา
        """
        raise NotImplementedError

class Node(Generic[S, A]):
    """
    "โหนด" (Node) ใน "แผนผังการค้นหา" (search tree)
    ประกอบด้วยตัวชี้ไปยัง "โหนดแม่" (parent node) (คือโหนดที่นำมายังโหนดนี้) และ "สถานะ" (state) จริงของโหนดนี้
    นอกจากนี้ยังรวมถึง "การกระทำ" (action) ที่ทำให้มาถึงสถานะนี้ และ "ต้นทุนเส้นทาง" (path_cost)
    ทั้งหมดจาก "โหนดราก" (root node) ไปยังโหนดนี้
    """
    def __init__(self, state: S, parent: Optional['Node'] = None, action: Optional[A] = None, path_cost: float = 0.0):
        """
        เริ่มต้น (initialize) โหนดใหม่ในแผนผังการค้นหา
        :param state: สถานะที่โหนดนี้แสดง
        :param parent: โหนดแม่ในแผนผังการค้นหา ถ้าเป็น None แสดงว่าเป็นโหนดราก
        :param action: การกระทำที่ทำจากโหนดแม่เพื่อมายังสถานะนี้ ถ้าเป็น None แสดงว่าเป็นโหนดราก
        :param path_cost: ต้นทุนสะสมจากโหนดรากมายังโหนดนี้
        """
        self.state: S = state
        self.parent: Optional['Node'] = parent
        self.action: Optional[A] = action
        self.path_cost: float = path_cost
        # คำนวณความลึกของโหนด โหนดรากมีความลึก 0
        self.depth: int = parent.depth + 1 if parent is not None else 0

    def __repr__(self) -> str:
        """
        คืนค่าการแสดงผลแบบสตริงของโหนด
        :return: การแสดงผลแบบสตริงที่รวมสถานะของโหนด
        """
        return f"<Node {self.state}>"

    def __lt__(self, other: 'Node') -> bool:
        """
        การเปรียบเทียบ "น้อยกว่า" (less-than) มีประโยชน์สำหรับการจัดการลำดับใน "คิวลำดับความสำคัญ" (priority queues)
        โดยจะเปรียบเทียบโหนดตามค่า path_cost ของแต่ละโหนด
        :param other: โหนดอื่นที่จะนำมาเปรียบเทียบ
        :return: True ถ้า path_cost ของโหนดนี้น้อยกว่าของโหนดอื่น, ไม่อย่างนั้นคืนค่า False
        """
        return self.path_cost < other.path_cost
    
    def expand(self, problem: Problem[S, A]) -> List['Node[S, A]']:
        """
        แสดงรายการโหนดที่สามารถเข้าถึงได้ในหนึ่งขั้นตอนจากโหนดนี้
        โดยจะสร้าง "โหนดลูก" (child nodes) ด้วยการใช้ "การกระทำ" (actions) ที่เป็นไปได้ทั้งหมดจากสถานะปัจจุบัน
        :param problem: อินสแตนซ์ของปัญหาการค้นหา
        :return: รายการของอ็อบเจกต์ Node ที่เป็นโหนดลูก
        """
        return [
            self.get_child_node(problem, action)
            for action in problem.actions(self.state)
        ]

    def get_child_node(self, problem: Problem[S, A], action: A) -> 'Node[S, A]':
        """
        สร้างและคืนค่า "โหนดลูก" (child node) ที่เกิดจากการทำ "การกระทำ" (action) ที่เฉพาะเจาะจง
        :param problem: อินสแตนซ์ของปัญหาการค้นหา
        :param action: การกระทำที่จะใช้เพื่อไปถึงสถานะลูก
        :return: อ็อบเจกต์ Node ใหม่ที่แสดงถึงสถานะลูก
        """
        next_state = problem.result(self.state, action)
        return Node(
            state=next_state,
            parent=self,
            action=action,
            # คำนวณ "ต้นทุนเส้นทาง" (path cost) ไปยังสถานะถัดไปโดยใช้เมธอด path_cost ของปัญหา
            path_cost=problem.path_cost(self.path_cost, self.state, action, next_state)
        )

    def solution(self) -> List[A]:
        """
        คืนค่าลำดับของ "การกระทำ" (actions) เพื่อไปจาก "โหนดราก" (root node) ไปยังโหนดนี้
        โดยการกระทำจะถูกดึงมาจากเส้นทาง โดยไม่รวมการกระทำที่นำไปสู่โหนดราก (ซึ่งเป็น None)
        :return: รายการของการกระทำที่แสดงถึงเส้นทางของโซลูชัน
        """
        return [node.action for node in self.path()[1:]]

    def path(self) -> List['Node[S, A]']:
        """
        คืนค่ารายการของ "โหนด" (nodes) ที่ประกอบกันเป็นเส้นทางจาก "โหนดราก" (root node) ไปยังโหนดนี้
        โดยจะย้อนรอยจากโหนดปัจจุบันกลับไปยังโหนดรากโดยใช้ตัวชี้ "โหนดแม่" (parent pointers)
        :return: รายการของอ็อบเจกต์ Node จากโหนดรากไปยังโหนดปัจจุบัน
        """
        node, path_back = self, []
        while node:
            path_back.append(node)
            node = node.parent
        return list(reversed(path_back))

    def __eq__(self, other) -> bool:
        """
        เปรียบเทียบโหนดสองโหนดว่าเท่ากันหรือไม่ โดยดูจาก "สถานะ" (state) ของแต่ละโหนด
        :param other: อ็อบเจกต์อื่นที่จะนำมาเปรียบเทียบ
        :return: True ถ้าอ็อบเจกต์อื่นเป็น Node และมีสถานะเหมือนกัน, ไม่อย่างนั้นคืนค่า False
        """
        return isinstance(other, Node) and self.state == other.state

    def __hash__(self) -> int:
        """
        คืนค่า "แฮช" (hash value) สำหรับโหนด โดยอ้างอิงจาก "สถานะ" (state) ของโหนด
        สิ่งนี้สำคัญสำหรับการใช้ Nodes ใน "เซ็ต" (sets) หรือเป็น "คีย์" (keys) ในพจนานุกรม (เช่น ในเซ็ต `explored`)
        :return: ค่าแฮชของสถานะของโหนด
        """
        return hash(self.state)

class Waitlist(Generic[S, A]):
    """
    คลาสพื้นฐาน (Abstract base class) สำหรับคิว (waitlists) ประเภทต่างๆ ที่ใช้ในอัลกอริทึมการค้นหา
    คลาสย่อยที่สืบทอดไปจะต้องใช้งานเมธอด put, get และ is_empty
    """
    def put(self, node: 'Node[S, A]', priority: Optional[float] = None):
        """
        เพิ่มโหนดลงใน waitlist
        :param node: โหนดที่จะเพิ่ม
        :param priority: ลำดับความสำคัญเสริมสำหรับโหนด (ใช้โดย PriorityQueue)
        """
        raise NotImplementedError
    def get(self) -> 'Node[S, A]':
        """
        ดึงและลบโหนดออกจาก waitlist
        :return: โหนดถัดไปจาก waitlist
        """
        raise NotImplementedError
    def is_empty(self) -> bool:
        """
        ตรวจสอบว่า waitlist ว่างเปล่าหรือไม่
        :return: True ถ้า waitlist ไม่มีโหนดอยู่เลย, ไม่อย่างนั้นคืนค่า False
        """
        raise NotImplementedError

class Stack(Waitlist):
    """
    การนำ Stack มาใช้งาน ซึ่งเป็นคิวแบบ LIFO (Last-In, First-Out)
    เหมาะสำหรับการค้นหาแบบ "เจาะลึก" (Depth-First Search หรือ DFS)
    """
    def __init__(self):
        """
        เริ่มต้น Stack ที่ว่างเปล่า
        """
        self._items: List['Node'] = []

    def put(self, node: 'Node[S, A]', priority: Optional[float] = None):
        """
        เพิ่มโหนดไปที่ด้านบนสุดของ Stack
        :param node: โหนดที่จะเพิ่ม
        :param priority: ไม่ได้ใช้ใน Stack แต่ยังคงมีไว้เพื่อให้เข้ากันได้กับอินเทอร์เฟซ
        """
        self._items.append(node)

    def get(self) -> 'Node[S, A]':
        """
        ลบและคืนค่าโหนดจากด้านบนสุดของ Stack
        :return: โหนดที่ถูกเพิ่มเข้ามาล่าสุด
        """
        return self._items.pop()

    def is_empty(self) -> bool:
        """
        ตรวจสอบว่า Stack ว่างเปล่าหรือไม่
        :return: True ถ้า Stack ไม่มีสมาชิกอยู่เลย, ไม่อย่างนั้นคืนค่า False
        """
        return len(self._items) == 0

class Queue(Waitlist):
    """
    การนำ Queue มาใช้งาน ซึ่งเป็นคิวแบบ FIFO (First-In, First-Out)
    เหมาะสำหรับการค้นหาแบบ "เจาะกว้าง" (Breadth-First Search หรือ BFS)
    """
    def __init__(self):
        """
        เริ่มต้น Queue ที่ว่างเปล่าโดยใช้ `deque` เพื่อให้การเพิ่มและลบจากทั้งสองด้านมีประสิทธิภาพ
        """
        self._items: Deque[Node] = deque()

    def put(self, node: 'Node[S, A]'):
        """
        เพิ่มโหนดไปที่ด้านหลังของคิว
        :param node: โหนดที่จะเพิ่ม
        """
        self._items.append(node)

    def get(self) -> 'Node[S, A]':
        """
        ลบและคืนค่าโหนดจากด้านหน้าของคิว
        :return: โหนดที่ถูกเพิ่มเข้ามานานที่สุด
        """
        return self._items.popleft()

    def is_empty(self) -> bool:
        """
        ตรวจสอบว่าคิวว่างเปล่าหรือไม่
        :return: True ถ้าคิวไม่มีสมาชิกอยู่เลย, ไม่อย่างนั้นคืนค่า False
        """
        return len(self._items) == 0

class PriorityQueue(Waitlist):
    """
    การนำ Priority Queue มาใช้งาน โดยใช้โมดูล `heapq`
    โหนดจะถูกจัดเรียงตาม "ลำดับความสำคัญ" (priority) โดยโหนดที่มีค่าลำดับความสำคัญต่ำสุดจะถูกดึงออกมาก่อน
    เหมาะสำหรับอัลกอริทึมการค้นหาแบบ "รู้ข้อมูล" (informed search)
    """
    def __init__(self):
        """
        เริ่มต้น PriorityQueue ที่ว่างเปล่า
        `_items` จะเก็บ tuple ที่ประกอบด้วย (priority, counter, node) เพื่อให้แน่ใจว่ามีการจัดเรียงที่เสถียร
        (FIFO สำหรับลำดับความสำคัญเดียวกัน) และการเปรียบเทียบที่ถูกต้องใน "ฮีป" (heap)
        `_count` ใช้เป็นตัวตัดสินเมื่อสมาชิกมีลำดับความสำคัญเท่ากัน
        """
        self._items: List[Tuple[float, int, 'Node[S, A]']] = []
        self._count: int = 0  # ใช้เพื่อให้มีการจัดเรียงที่เสถียรสำหรับสมาชิกที่มีลำดับความสำคัญเท่ากัน

    def put(self, node: 'Node[S, A]', priority: Optional[float] = None):
        """
        เพิ่มโหนดเข้าสู่ Priority Queue
        หากไม่ได้กำหนด priority ไว้ จะใช้ path_cost ของโหนดเป็นค่าเริ่มต้น (มีประโยชน์สำหรับ UCS)
        :param node: โหนดที่จะเพิ่ม
        :param priority: ค่าลำดับความสำคัญสำหรับโหนด ค่าที่น้อยกว่าหมายถึงลำดับความสำคัญสูงกว่า
        """
        if priority is None:
            priority = node.path_cost
        # เพิ่ม tuple (priority, count, node) ลงในฮีป
        # ตัวแปร count ช่วยให้แน่ใจว่าถ้าโหนดสองโหนดมีลำดับความสำคัญเท่ากัน
        # โหนดที่ถูกเพิ่มเข้าไปก่อนจะถูกดึงออกมาก่อน (FIFO tie-breaking)
        heapq.heappush(self._items, (priority, self._count, node))
        self._count += 1

    def get(self) -> 'Node[S, A]':
        """
        ลบและคืนค่าโหนดที่มีลำดับความสำคัญต่ำที่สุดจากคิว
        :return: โหนดที่มีลำดับความสำคัญสูงสุด (ค่าลำดับความสำคัญต่ำสุด)
        """
        priority, count, node = heapq.heappop(self._items)
        self._count -= 1 # ลดค่า count เพื่อสะท้อนการลบออก แม้จะไม่จำเป็นสำหรับ logic ของ _count
        return node

    def is_empty(self) -> bool:
        """
        ตรวจสอบว่า Priority Queue ว่างเปล่าหรือไม่
        :return: True ถ้าคิวไม่มีสมาชิกอยู่เลย, ไม่อย่างนั้นคืนค่า False
        """
        return self._count == 0 # หรือจะใช้ return len(self._items) == 0 ก็ได้

# --- อัลกอริทึมการค้นหา ---
def uninform_search(problem: Problem, waitlist:Waitlist, initial: S, goal: S, verbose: bool = False) -> Optional[Node]:
    """
    อัลกอริทึมการค้นหาแบบ "ไม่รู้ข้อมูล" (uninformed search) ทั่วไป
    สามารถทำงานเป็น "Breadth-First Search (BFS)" หาก `waitlist` เป็น Queue,
    หรือ "Depth-First Search (DFS)" หาก `waitlist` เป็น Stack
    อัลกอริทึมนี้จะสำรวจโหนดใน "แผนผังการค้นหา" (search tree) ตามลำดับที่กำหนดโดย waitlist
    :param problem: อินสแตนซ์ของปัญหาการค้นหา
    :param waitlist: อินสแตนซ์ของ Waitlist (เช่น Queue สำหรับ BFS, Stack สำหรับ DFS)
    :param initial: สถานะเริ่มต้นของปัญหา
    :param goal: สถานะเป้าหมายของปัญหา
    :param verbose: ถ้าเป็น True จะพิมพ์ข้อมูลกระบวนการค้นหาโดยละเอียด
    :return: โหนดเป้าหมาย (goal node) หากพบโซลูชัน, ไม่อย่างนั้นคืนค่า None
    """
    # เริ่มต้นด้วยสถานะแรกสุดเป็นโหนดราก
    waitlist.put(Node(initial))
    # เก็บ "สถานะที่สำรวจแล้ว" (explored states) เพื่อหลีกเลี่ยงการวนซ้ำ (cycles) และการคำนวณซ้ำซ้อน
    explored: Set[S] = set()
    count = 0 # ตัวนับสำหรับขั้นตอน/โหนดที่ประมวลผล

    while not waitlist.is_empty(): # ทำต่อตราบใดที่ยังมีโหนดให้สำรวจ
        current_node = waitlist.get() # ดึงโหนดถัดไปจาก waitlist (BFS: โหนดที่ตื้นที่สุด, DFS: โหนดที่ลึกที่สุด)
        count += 1
        if verbose: print(f"{count}. โหนดปัจจุบัน: {current_node}, สถานะ: {current_node.state}")

        # ตรวจสอบว่าสถานะของโหนดปัจจุบันเป็นสถานะเป้าหมายหรือไม่
        if problem.goal_test(current_node.state, goal):
            if verbose: print(f"พบเป้าหมาย: {current_node}")
            return current_node # คืนค่าโหนดเป้าหมาย ซึ่งมีเส้นทางไปสู่โซลูชัน

        # ขยายโหนดปัจจุบัน และเพิ่มสถานะลูกใหม่ที่ยังไม่เคยสำรวจเข้าไปใน waitlist
        if current_node.state not in explored:
            explored.add(current_node.state) # ทำเครื่องหมายสถานะใหม่ว่าสำรวจแล้ว
            if verbose: print(f"  -> เพิ่มไปยัง explored: {current_node.state}")
            for child in current_node.expand(problem):     
                waitlist.put(child) # เพิ่มโหนดลูกเข้าไปใน waitlist เพื่อสำรวจต่อไป
                if verbose: print(f"  -> เพิ่มไปยัง waitlist: {child.state}")
            if verbose: print(f" :: waitlist ปัจจุบัน: {waitlist}")

    if verbose: print("ไม่พบโซลูชัน")
    return None # ถ้า waitlist ว่างเปล่าและไม่พบเป้าหมาย แสดงว่าไม่มีโซลูชัน

def breadth_first_search(problem: Problem, initial: S, goal: S, verbose: bool = False) -> Optional[Node]:
    """
    อัลกอริทึม Breadth-First Search
    เป็นการค้นหาแบบ กว้างก่อน ใช้ Queue ในการเก็บโหนดรอสำรวจ
    :param problem: อินสแตนซ์ของปัญหาการค้นหา
    :param initial: สถานะเริ่มต้นของปัญหา
    :param goal: สถานะเป้าหมายของปัญหา
    :param verbose: ถ้าเป็น True จะพิมพ์ข้อมูลกระบวนการค้นหาโดยละเอียด
    :return: โหนดเป้าหมาย (goal node) หากพบโซลูชัน, ไม่อย่างนั้นคืนค่า None
    """
    return uninform_search(
        problem,
        Queue(),
        initial,
        goal,
        verbose=verbose)
def depth_first_search(problem: Problem, initial: S, goal: S, verbose: bool = False) -> Optional[Node]:
    """
    อัลกอริทึม Depth-First Search
    เป็นการค้นหาแบบ ลึกก่อน ใช้ stack ในการเก็บโหนดรอสำรวจ
    :param problem: อินสแตนซ์ของปัญหาการค้นหา
    :param initial: สถานะเริ่มต้นของปัญหา
    :param goal: สถานะเป้าหมายของปัญหา
    :param verbose: ถ้าเป็น True จะพิมพ์ข้อมูลกระบวนการค้นหาโดยละเอียด
    :return: โหนดเป้าหมาย (goal node) หากพบโซลูชัน, ไม่อย่างนั้นคืนค่า None
    """
    return uninform_search(
        problem,
        Stack(),
        initial,
        goal,
        verbose=verbose)

def best_first_search(problem: Problem, initial: S, goal: S, f: Callable[[Node, S], float], verbose: bool = False) -> Optional[Node]:
    """
    อัลกอริทึม Best-First Search
    ค้นหากราฟโดยการขยาย "โหนดที่มีแนวโน้มมากที่สุด" (most promising node)
    ที่เลือกตามฟังก์ชันประเมินผล `f` ที่กำหนดไว้
    โดยจะใช้ PriorityQueue ในการจัดการลำดับการสำรวจ
    นี่คือโครงสร้างทั่วไปสำหรับอัลกอริทึมการค้นหาแบบ "รู้ข้อมูล" (informed search) เช่น UCS, Greedy BFS, A*
    :param problem: อินสแตนซ์ของปัญหาการค้นหา
    :param initial: สถานะเริ่มต้นของปัญหา
    :param goal: สถานะเป้าหมายของปัญหา
    :param f: ฟังก์ชันประเมินผล f(node, goal) ที่คืนค่าลำดับความสำคัญเป็นตัวเลข
              ค่า f ที่ต่ำกว่าหมายถึงลำดับความสำคัญสูงกว่า
    :param verbose: ถ้าเป็น True จะพิมพ์ข้อมูลกระบวนการค้นหาโดยละเอียด
    :return: โหนดเป้าหมาย (goal node) หากพบโซลูชัน, ไม่อย่างนั้นคืนค่า None
    """
    node = Node(initial) # สร้างโหนดเริ่มต้น
    waitlist = PriorityQueue() # ใช้ PriorityQueue สำหรับการเลือกแบบ best-first
    waitlist.put(node, f(node, goal)) # เพิ่มโหนดเริ่มต้นโดยใช้ค่า f เป็นลำดับความสำคัญ
    explored: Set[S] = set() # เก็บ "สถานะที่สำรวจแล้ว" (explored states)
    count = 0 # ตัวนับสำหรับขั้นตอน/โหนดที่ประมวลผล

    while not waitlist.is_empty():
        current_node = waitlist.get() # ดึงโหนดที่มีค่า f ต่ำที่สุด (ลำดับความสำคัญสูงสุด)
        count += 1
        if verbose: print(f"{count}. โหนดปัจจุบัน: {current_node}, f(n)={f(current_node, goal)}")
        
        # ตรวจสอบว่าสถานะของโหนดปัจจุบันเป็นสถานะเป้าหมายหรือไม่
        if problem.goal_test(current_node.state, goal):
            if verbose: print(f"พบเป้าหมาย: {current_node}")
            return current_node

        # ถ้าสถานะปัจจุบันถูกสำรวจไปแล้ว ให้ข้ามไป (จัดการกรณีที่มีการเพิ่มโหนดซ้ำด้วยต้นทุนที่สูงกว่า,หากมี replace ไม่จำเป็นต้องมีส่วนนี้)
        if current_node.state in explored:
            continue
            
        explored.add(current_node.state) # ทำเครื่องหมายสถานะปัจจุบันว่าสำรวจแล้ว
        
        # ขยายโหนดปัจจุบันและเพิ่มสถานะลูกใหม่ที่ยังไม่เคยสำรวจเข้าไปใน waitlist
        for child in current_node.expand(problem):
            # เพิ่มเข้าไปใน waitlist เฉพาะเมื่อสถานะของลูกยังไม่ได้รับการสำรวจอย่างสมบูรณ์เท่านั้น
            # โหนดที่อยู่ใน waitlist แล้วซึ่งอาจมีต้นทุนสูงกว่า อาจถูกเพิ่มใหม่หากพบเส้นทางที่ดีกว่า
            # แต่การตรวจสอบนี้จะป้องกันการเพิ่มสถานะที่ "ประมวลผลแล้ว" ซ้ำ
            if child.state not in explored: # นี่คือความแตกต่างสำคัญจาก uniform_cost_search เพื่อความเหมาะสมในกราฟบางประเภท
                waitlist.put(child, f(child, goal)) # เพิ่มโหนดลูกด้วยค่า f ที่คำนวณได้
                if verbose: print(f"  -> เพิ่มไปยัง waitlist: {child.state} ด้วย f(n)={f(child, goal)}")
                                
        if verbose: print(f" :: waitlist ปัจจุบัน: {waitlist}")
    
    if verbose: print("ไม่พบโซลูชัน")
    return None

def greedy_best_first_search(problem: Problem, initial: S, goal: S, verbose: bool = False) -> Optional[Node]:
    """
    อัลกอริทึม Greedy Best-First Search
    เป็นการค้นหาแบบ Best-First Search ที่ใช้แค่ "ฮิวริสติก" h(n) เป็นฟังก์ชันประเมินผล `f(n) = h(n)`
    โดยจะให้ความสำคัญกับโหนดที่ดูเหมือนจะใกล้เป้าหมายที่สุด ตามการประมาณค่าของฮิวริสติก
    :param problem: อินสแตนซ์ของปัญหาการค้นหา
    :param initial: สถานะเริ่มต้นของปัญหา
    :param goal: สถานะเป้าหมายของปัญหา
    :param verbose: ถ้าเป็น True จะพิมพ์ข้อมูลกระบวนการค้นหาโดยละเอียด
    :return: โหนดเป้าหมาย (goal node) หากพบโซลูชัน, ไม่อย่างนั้นคืนค่า None
    """
    return best_first_search(
        problem,
        initial,
        goal,
        f=lambda node, g: problem.heuristic(node.state, g), # ฟังก์ชันประเมินผลคือแค่ heuristic
        verbose=verbose
    )

def uniform_cost_search(problem: Problem, initial: S, goal: S, verbose: bool = False) -> Optional[Node]:
    """
    อัลกอริทึม Uniform Cost Search (UCS)
    เป็นการค้นหาแบบ Best-First Search ที่ใช้แค่ "ต้นทุนเส้นทาง" g(n) เป็นฟังก์ชันประเมินผล `f(n) = g(n)`
    โดยจะขยายโหนดที่มีต้นทุนเส้นทางต่ำที่สุดจากโหนดเริ่มต้น ซึ่งรับประกันว่าจะได้โซลูชันที่ดีที่สุด (optimal)
    หากต้นทุนของขอบ (edge costs) ไม่เป็นค่าลบ
    :param problem: อินสแตนซ์ของปัญหาการค้นหา
    :param initial: สถานะเริ่มต้นของปัญหา
    :param goal: สถานะเป้าหมายของปัญหา
    :param verbose: ถ้าเป็น True จะพิมพ์ข้อมูลกระบวนการค้นหาโดยละเอียด
    :return: โหนดเป้าหมาย (goal node) หากพบโซลูชัน, ไม่อย่างนั้นคืนค่า None
    """
    return best_first_search(
        problem,
        initial,
        goal,
        f=lambda node, g: node.path_cost, # ฟังก์ชันประเมินผลคือแค่ path cost
        verbose=verbose
    )

def astar_search(problem: Problem, initial: S, goal: S, verbose: bool = False) -> Optional[Node]:
    """
    อัลกอริทึม A* Search
    เป็นการค้นหาแบบ Best-First Search ที่ใช้ฟังก์ชันประเมินผล `f(n) = g(n) + h(n)`
    โดยที่ `g(n)` คือ "ต้นทุนเส้นทาง" (path cost) จากโหนดเริ่มต้นมายังโหนด n, และ `h(n)` คือ
    "ฮิวริสติก" (heuristic) (ค่าประมาณต้นทุนจากโหนด n ไปยังเป้าหมาย)
    A* Search เหมาะสม (optimal) และสมบูรณ์ (complete) หากฮิวริสติกสามารถยอมรับได้ (admissible) (สำหรับการค้นหาแบบ tree search)
    หรือยอมรับได้และสอดคล้องกัน (consistent) (สำหรับการค้นหาแบบ graph search)
    :param problem: อินสแตนซ์ของปัญหาการค้นหา
    :param initial: สถานะเริ่มต้นของปัญหา
    :param goal: สถานะเป้าหมายของปัญหา
    :param verbose: ถ้าเป็น True จะพิมพ์ข้อมูลกระบวนการค้นหาโดยละเอียด
    :return: โหนดเป้าหมาย (goal node) หากพบโซลูชัน, ไม่อย่างนั้นคืนค่า None
    """
    return best_first_search(
        problem,
        initial,
        goal,
        f=lambda node, g: node.path_cost + problem.heuristic(node.state, g), # f(n) = g(n) + h(n)
        verbose=verbose
    )

"""
Constraint Satisfaction Problem 
โมดูลส่วนขยายสำหรับปัญหา Constraint Satisfaction Problem
โดยใช้อัลกอริทึม backtracking search ในการแก้ปัญหา
"""

class CSPState:
    """แสดงสถานะของการกำหนดค่าให้กับตัวแปรใน CSP"""
    def __init__(self, assignment: Dict[str, Any] = None):
        self.assignment = assignment or {}
    
    def __eq__(self, other) -> bool:
        return isinstance(other, CSPState) and self.assignment == other.assignment
    
    def __hash__(self) -> int:
        return hash(tuple(sorted(self.assignment.items())))
    
    def __repr__(self) -> str:
        return f"CSPState({self.assignment})"

class CSPAction:
    """แสดงการกระทำในการกำหนดค่าให้กับตัวแปร"""
    def __init__(self, variable: str, value: Any):
        self.variable = variable
        self.value = value
    
    def __eq__(self, other) -> bool:
        return (isinstance(other, CSPAction) and 
                self.variable == other.variable and 
                self.value == other.value)
    
    def __hash__(self) -> int:
        return hash((self.variable, self.value))
    
    def __repr__(self) -> str:
        return f"CSPAction({self.variable}={self.value})"

class CSPProblem(Problem[CSPState, CSPAction]):
    """
    Framework สำหรับปัญหา Constraint Satisfaction Problem ที่สืบทอดมาจาก Problem
    คลาสนี้จัดเตรียมโครงสร้างสำหรับปัญหา CSP ที่มีตัวแปร, domain และ constraint
    """
    
    def __init__(self, variables: List[str], domains: Dict[str, List[Any]]):
        """
        เริ่มต้นปัญหา CSP ด้วยตัวแปรและ domain ของตัวแปร
        
        Args:
            variables: รายการชื่อตัวแปร
            domains: Dictionary ที่เชื่อมโยงตัวแปรกับค่าที่เป็นไปได้
        """
        self.variables = variables
        self.domains = domains
        self.initial_state = CSPState()
    
    def actions(self, state: CSPState) -> List[CSPAction]:
        """
        คืนค่า action ที่เป็นไปได้สำหรับสถานะปัจจุบัน
        เป็นการ implement method ที่จำเป็นจาก Problem base class
        
        Args:
            state: สถานะ CSP ปัจจุบันพร้อมการกำหนดค่าตัวแปร
            
        Returns:
            รายการ CSP action ที่เป็นไปได้ (การกำหนดค่าตัวแปร)
        """
        unassigned_var = self.select_unassigned_variable(state)
        if unassigned_var is None:
            return []
        
        return [CSPAction(unassigned_var, value) 
                for value in self.ordered_domain_values(unassigned_var, state)]
    
    def result(self, state: CSPState, action: CSPAction) -> CSPState:
        """
        นำ action มาใช้เพื่อสร้างสถานะใหม่
        เป็นการ implement method ที่จำเป็นจาก Problem base class
        
        Args:
            state: สถานะ CSP ปัจจุบัน
            action: การกระทำที่จะนำมาใช้ (การกำหนดค่าตัวแปร)
            
        Returns:
            สถานะ CSP ใหม่ที่มีการกำหนดค่าตัวแปรแล้ว
        """
        new_assignment = state.assignment.copy()
        new_assignment[action.variable] = action.value
        return CSPState(new_assignment)
    
    def goal_test(self, state: CSPState, goal: CSPState = None) -> bool:
        """
        ตรวจสอบว่าสถานะปัจจุบันเป็นคำตอบที่สมบูรณ์หรือไม่
        เป็นการ implement method ที่จำเป็นจาก Problem base class
        
        Args:
            state: สถานะ CSP ปัจจุบันที่จะทดสอบ
            goal: สถานะเป้าหมาย (ไม่ใช้ใน CSP, ค่าเริ่มต้นเป็น None)
            
        Returns:
            True ถ้าตัวแปรทั้งหมดได้รับการกำหนดค่าและ constraint ทั้งหมดเป็นจริง
        """
        if len(state.assignment) != len(self.variables):
            return False
        
        # ตรวจสอบว่า constraint ทั้งหมดเป็นจริงหรือไม่
        return self.is_complete_assignment_consistent(state.assignment)
    
    def path_cost(self, current_cost: float, state1: CSPState, 
                  action: CSPAction, state2: CSPState) -> float:
        """
        คำนวณ path cost สำหรับ CSP (ค่าคงที่ 1 ต่อการกำหนดค่า)
        เป็นการ implement method ที่จำเป็นจาก Problem base class
        
        Args:
            current_cost: ค่าใช้จ่ายที่สะสมปัจจุบัน
            state1: สถานะก่อนหน้า
            action: การกระทำที่ดำเนินการ
            state2: สถานะผลลัพธ์
            
        Returns:
            path cost ที่อัปเดตแล้ว
        """
        return current_cost + 1
    
    def heuristic(self, state: CSPState, goal: CSPState = None) -> float:
        """
        ฟังก์ชัน heuristic สำหรับ CSP (จำนวนตัวแปรที่ยังไม่ได้กำหนดค่า)
        เป็นการ implement method ที่จำเป็นจาก Problem base class
        
        Args:
            state: สถานะ CSP ปัจจุบัน
            goal: สถานะเป้าหมาย (ไม่ใช้ใน CSP)
            
        Returns:
            ค่า heuristic (จำนวนตัวแปรที่เหลือต้องกำหนดค่า)
        """
        return len(self.variables) - len(state.assignment)
    
    def select_unassigned_variable(self, state: CSPState) -> Optional[str]:
        """
        เลือกตัวแปรถัดไปที่จะกำหนดค่าโดยใช้ Most Remaining Values heuristic
        
        Args:
            state: สถานะ CSP ปัจจุบัน
            
        Returns:
            ชื่อตัวแปรที่จะกำหนดค่าถัดไป หรือ None ถ้าตัวแปรทั้งหมดได้รับการกำหนดค่าแล้ว
        """
        unassigned = [var for var in self.variables if var not in state.assignment]
        if not unassigned:
            return None
        return min(unassigned, key=lambda var: len(self.get_remaining_values(var, state)))
    
    def get_remaining_values(self, variable: str, state: CSPState) -> List[Any]:
        """
        หาค่าที่เหลือและเป็นไปได้สำหรับตัวแปรจากการกำหนดค่าปัจจุบัน
        
        Args:
            variable: ชื่อตัวแปร
            state: สถานะ CSP ปัจจุบัน
            
        Returns:
            รายการค่าที่เป็นไปได้สำหรับตัวแปร
        """
        return [value for value in self.domains[variable] 
                if self.is_consistent(variable, value, state.assignment)]
    
    def ordered_domain_values(self, variable: str, state: CSPState) -> List[Any]:
        """
        คืนค่า domain ที่เรียงลำดับตามความสอดคล้องกับ constraint
        
        Args:
            variable: ชื่อตัวแปร
            state: สถานะ CSP ปัจจุบัน
            
        Returns:
            รายการค่าที่เป็นไปได้สำหรับตัวแปร
        """
        return self.get_remaining_values(variable, state)
    
    def is_consistent(self, variable: str, value: Any, assignment: Dict[str, Any]) -> bool:
        """
        ตรวจสอบว่าการกำหนดค่าให้กับตัวแปรสอดคล้องกับการกำหนดค่าปัจจุบันหรือไม่
        method นี้ต้องได้รับการ implement โดย subclass
        
        Args:
            variable: ชื่อตัวแปร
            value: ค่าที่จะกำหนด
            assignment: การกำหนดค่าตัวแปรปัจจุบัน
            
        Returns:
            True ถ้าการกำหนดค่าสอดคล้องกับ constraint
        """
        raise NotImplementedError("Subclass ต้อง implement การตรวจสอบ constraint")
    
    def is_complete_assignment_consistent(self, assignment: Dict[str, Any]) -> bool:
        """
        ตรวจสอบว่าการกำหนดค่าที่สมบูรณ์เป็นไปตาม constraint ทั้งหมดหรือไม่
        
        Args:
            assignment: การกำหนดค่าตัวแปรที่สมบูรณ์
            
        Returns:
            True ถ้าการกำหนดค่าเป็นไปตาม constraint ทั้งหมด
        """
        for variable in self.variables:
            if not self.is_consistent(variable, assignment[variable], assignment):
                return False
        return True

def backtracking_search(problem: CSPProblem, verbose: bool = False) -> Optional[Dict[str, Any]]:
    """
    แก้ปัญหา CSP โดยใช้อัลกอริทึม backtracking search 
    
    Args:
        problem: instance ของปัญหา CSP
        verbose: เปิดใช้งานการแสดงผลรายละเอียด
        
    Returns:
        Dictionary ของการกำหนดค่าตัวแปรถ้าพบคำตอบ หรือ None ถ้าไม่พบ
    """
    
    class BacktrackingWaitlist:
        """waitlist แบบกำหนดเองที่ implement พฤติกรรม backtracking"""
        def __init__(self):
            self._stack = []
        
        def put(self, node: Node, priority: Optional[float] = None):
            self._stack.append(node)
        
        def get(self) -> Node:
            return self._stack.pop()
        
        def is_empty(self) -> bool:
            return len(self._stack) == 0
    
    def recursive_backtracking(node: Node[CSPState, CSPAction], depth: int = 0) -> Optional[Node]:
        """
        การ implement backtracking แบบ recursive โดยใช้โครงสร้าง Node 
        
        Args:
            node: node ปัจจุบันใน search tree
            depth: ความลึกปัจจุบันสำหรับการแสดงผล
            
        Returns:
            solution node ถ้าพบคำตอบ หรือ None ถ้าไม่พบ
        """
        if verbose:
            indent = "  " * depth
            assigned_count = len(node.state.assignment)
            total_vars = len(problem.variables)
            print(f"{indent}กำลังสำรวจ node: {assigned_count}/{total_vars} ตัวแปรได้รับการกำหนดค่า")
        
        # ตรวจสอบว่าเรามีคำตอบที่สมบูรณ์หรือไม่
        if problem.goal_test(node.state):
            if verbose:
                print(f"{indent}พบคำตอบแล้ว!")
            return node
        
        # หา action ที่เป็นไปได้ (การกำหนดค่าตัวแปร)
        actions = problem.actions(node.state)
        if not actions:
            if verbose:
                print(f"{indent}ไม่มี action ที่เป็นไปได้")
            return None
        
        # ลอง action แต่ละตัว
        for action in actions:
            if verbose:
                print(f"{indent}กำลังลอง {action.variable} = {action.value}")
            
            # สร้าง child node
            child_node = node.get_child_node(problem, action)
            
            # ค้นหาแบบ recursive จาก child นี้
            result = recursive_backtracking(child_node, depth + 1)
            if result is not None:
                return result
            
            if verbose:
                print(f"{indent}Backtracking จาก {action.variable} = {action.value}")
        
        if verbose:
            print(f"{indent}ไม่พบคำตอบสำหรับ path ปัจจุบัน")
        return None
    
    # เริ่มต้นด้วย initial state
    initial_node = Node(problem.initial_state)
    solution_node = recursive_backtracking(initial_node)
    
    if solution_node:
        return solution_node.state.assignment
    return None

def csp_search(problem: CSPProblem, search_algorithm: str = "backtracking", 
                          verbose: bool = False) -> Optional[Dict[str, Any]]:
    """
    แก้ปัญหา CSP โดยใช้อัลกอริทึมการค้นหาต่างๆ
    
    Args:
        problem: instance ของปัญหา CSP
        search_algorithm: อัลกอริทึมการค้นหาที่จะใช้ ("backtracking", "dfs", "best_first")
        verbose: เปิดใช้งานการแสดงผลรายละเอียด
        
    Returns:
        Dictionary ของการกำหนดค่าตัวแปรถ้าพบคำตอบ หรือ None ถ้าไม่พบ
    """
    initial_state = problem.initial_state
    
    # สำหรับความเข้ากันได้กับ goal_test เราสร้าง dummy goal state
    # การทดสอบเป้าหมายจริงจะถูกจัดการโดย goal_test method ของปัญหา
    goal_state = CSPState()
    
    if search_algorithm == "backtracking":
        return backtracking_search(problem, verbose)
    
    elif search_algorithm == "dfs":
        result_node = depth_first_search(problem, initial_state, goal_state, verbose)
        return result_node.state.assignment if result_node else None
    
    elif search_algorithm == "best_first":
        result_node = best_first_search(
            problem, 
            initial_state, 
            goal_state,
            f=lambda node, goal: problem.heuristic(node.state, goal),
            verbose=verbose
        )
        return result_node.state.assignment if result_node else None
    
    else:
        raise ValueError(f"อัลกอริทึมการค้นหาไม่รู้จัก: {search_algorithm}")

# ส่วนตัวอย่างการใช้งาน
if __name__ == '__main__':
    # 1. กำหนดปัญหา "กราฟ" (graph) แบบง่ายๆ
    class SimpleGraphProblem(Problem[str, str]):
        """
        เป็นการนำคลาส Problem มาใช้งานจริง สำหรับ "กราฟมีทิศทาง" (directed graph) แบบง่าย
        "สถานะ" (states) จะถูกแทนด้วยสตริง (ชื่อโหนด) และ "การกระทำ" (actions) ก็เป็นสตริงเช่นกัน
        ซึ่งแสดงถึงโหนดปลายทาง
        """
        def __init__(self, graph):
            """
            เริ่มต้น SimpleGraphProblem
            :param graph: พจนานุกรม (dictionary) ที่แสดงถึงกราฟ ตัวอย่างเช่น
                          `{'A': {'B': 1, 'C': 4}, 'B': {'C': 2, 'D': 5}, ...}`
                          โดยที่คีย์คือ "โหนด" (nodes) และค่าคือพจนานุกรมที่แมป
                          "โหนดเพื่อนบ้าน" (neighboring nodes) กับ "ต้นทุน" (cost) ของ "ขอบ" (edge)
            """
            self.graph = graph

        def actions(self, state: str) -> List[str]:
            """
            คืนค่ารายการของ "โหนดปลายทาง" (destination nodes) ที่เป็นไปได้ (ซึ่งคือ "การกระทำ")
            จาก "สถานะ" (state) ปัจจุบัน
            :param state: โหนดปัจจุบันในกราฟ
            :return: รายการของสตริง ซึ่งแต่ละสตริงแสดงถึงโหนดเพื่อนบ้านที่สามารถไปถึงได้
            """
            return list(self.graph.get(state, {}).keys())

        def result(self, state: str, action: str) -> str:
            """
            คืนค่า "สถานะ" (state) ที่เป็นผลจากการทำ "การกระทำ" (action) จากสถานะปัจจุบัน
            ในกราฟอย่างง่ายนี้ การกระทำคือสถานะถัดไปโดยตรง
            :param state: โหนดปัจจุบัน
            :param action: การกระทำที่จะทำ ซึ่งก็คือชื่อของโหนดถัดไป
            :return: ชื่อของโหนดถัดไป
            """
            return action
        
        def path_cost(self, current_cost: float, state1: str, action: str, state2: str) -> float:
            """
            คืนค่า "ต้นทุน" (cost) ของการย้ายจาก state1 ไปยัง state2 โดยทำ "การกระทำ" (action)
            แล้วนำไปบวกกับ `current_cost`
            ต้นทุนจะถูกดึงมาจากข้อมูลกราฟโดยตรง
            :param current_cost: ต้นทุนสะสมที่ใช้ไปเพื่อไปถึง state1
            :param state1: โหนดเริ่มต้นของขอบ (edge)
            :param action: การกระทำที่ทำ (ซึ่งในที่นี้คือ state2 ในการแสดงกราฟนี้)
            :param state2: โหนดปลายทางของขอบ (edge)
            :return: ต้นทุนรวมในการไปถึง state2 จากจุดเริ่มต้น
            """
            # พารามิเตอร์ `action` ในปัญหานี้แสดงถึง `state2` โดยตรง ดังนั้นเราจึงสามารถใช้ `graph[state1][state2]` ได้
            return current_cost + self.graph[state1][state2]

        def heuristic(self, state: S, goal: S) -> float:
            """
            "ฮิวริสติก" (heuristic) อย่างง่ายสำหรับปัญหา "กราฟ" (graph) นี้
            สำหรับตัวอย่างนี้ เราจะใช้ฮิวริสติกเป็น 0
            ซึ่งจะทำให้ A* ทำงานเหมือน Uniform Cost Search หากไม่มีการกำหนดค่าอื่น
            ในสถานการณ์จริง ควรเป็นค่าประมาณระยะทางเส้นตรง หรืออื่นๆ
            :param state: โหนดปัจจุบัน
            :param goal: โหนดเป้าหมาย
            :return: ค่าประมาณฮิวริสติกจากสถานะไปยังเป้าหมาย
            """
            # สำหรับกราฟทั่วไปที่ไม่มีการระบุพิกัด ฮิวริสติกที่ยอมรับได้ทั่วไปคือ 0
            # ซึ่งจะทำให้ A* กลายเป็นการค้นหาแบบ Uniform Cost Search
            # ในการทำให้ A* มีประสิทธิภาพจริงๆ จำเป็นต้องมีฮิวริสติกที่เฉพาะเจาะจงและ "รู้ข้อมูล" (informed) สำหรับปัญหา
            return 0.0

    # 2. สร้างอินสแตนซ์ของปัญหา
    # กำหนดกราฟอย่างง่ายพร้อมโหนดและต้นทุนของขอบ
    simple_graph = {
        'A': {'B': 1, 'C': 4},
        'B': {'C': 2, 'D': 5},
        'C': {'D': 1},
        'D': {} # D เป็นโหนดสุดท้าย ไม่มี "การกระทำ" (actions) จากโหนดนี้
    }
    problem = SimpleGraphProblem(simple_graph)
    initial_state = 'A'
    goal_state = 'D'

    # 3. รันอัลกอริทึมการค้นหา
    print("กำลังรัน A-Star Search...")
    # ตั้งค่า verbose=True เพื่อดูขั้นตอนของกระบวนการค้นหาโดยละเอียด
    solution_node = astar_search(problem, initial_state, goal_state, verbose=True)

    # 4. พิมพ์โซลูชัน
    if solution_node:
        print(f"\nพบเส้นทาง: {' -> '.join(n.state for n in solution_node.path())}")
        print(f"การกระทำของโซลูชัน: {solution_node.solution()}")
        print(f"ต้นทุนรวม: {solution_node.path_cost}")
    else:
        print("ไม่พบโซลูชัน")