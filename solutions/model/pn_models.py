from collections import deque
from dataclasses import dataclass, field
from time import process_time
from typing import Deque, List, Optional, Tuple

@dataclass
class Token:
    job_id: int
    enter_time: int
    wafer_type: int = -1
    deadline: int = None

    def clone(self) -> "Token":
        return Token(job_id=self.job_id, enter_time=self.enter_time,wafer_type=self.wafer_type)


@dataclass
class Place:
    name: str
    capacity: int
    processing_time: int
    type: int  # 1 for manipulator place, 2 for delivery place, 3 for idle place, 4 for source place


    tokens: Deque[Token] = field(default_factory=deque)

    def clone(self) -> "Place":
        cloned = Place(name=self.name, capacity=self.capacity, processing_time=self.processing_time,type=self.type)
        cloned.tokens = deque(tok.clone() for tok in self.tokens)
        return cloned

    def head(self) -> Token:
        return self.tokens[0]

    def pop_head(self) -> Token:
        return self.tokens.popleft()

    def remove(self):
        token = self.pop_head()
        if self.type in [1,2,4]:
            return token.job_id
        else:
            return None

    def append(self, token: Token) -> None:
        self.tokens.append(token)

    def res_time(self, current_time: int) -> int:
        """返回当前库所内wafer的剩余超时时间"""
        if len(self.tokens) == 0:
            return 10**5
        else:
            # Type 1 (p_i): Process chamber places with processing time constraint
            # Type 2 (d_i): Delivery/transport places with 30s transport constraint
            if self.type == 1:  # Process chamber (p_i)
                res_time = self.head().enter_time + self.processing_time + 20 - current_time
            elif self.type == 2:  # Transport place (d_i)
                res_time = self.head().enter_time + 30 - current_time
            else:
                # Type 3 (idle), Type 4 (other) - no deadline
                return 10**5

            return -1 if res_time < 0 else int(res_time)

    def __len__(self) -> int:
        return len(self.tokens)
