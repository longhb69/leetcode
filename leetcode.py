def is_paired(input_string):
    balanced = True
    brackets = '[]'
    braces = '{}'
    parentheses = '()'

    filter_string = [char for char in input_string if char in brackets or char in braces or char in parentheses]
    print(filter_string)
    i = 0
    length = len(filter_string)
    end = length - 1
    count = 0
    while(count != length and i < length):
        close = ''
        if filter_string[i] in brackets:
            close = brackets[1]
        elif filter_string[i] in braces:
            close = braces[1]
        elif filter_string[i] in parentheses:
            close = parentheses[1]
        
        if i+1 < length and filter_string[i+1] == close:
            print(filter_string[i], filter_string[i+1])
            i += 2
        else:
            if i < length:
                print("check: ", filter_string[i])
                if filter_string[i] == close:
                    balanced = False
                    break
                if filter_string[end] == close:
                    print(filter_string[i], filter_string[end])
                    i += 1
                    end -= 1
                else:
                    balanced = False
                    break
        count += 2

    return balanced

def is_paired_v2(input_string):
    brackets = {']': '[', '}' : '{', ')': '('}
    stack = []
    for ch in input_string:
        if ch in brackets.values():
            stack.append(ch)
        elif ch in brackets.keys():
            if len(stack) == 0:
                return False
            b = stack.pop()
            if brackets[ch] != b:
                return False
    return len(stack) == 0;


def factors(value):
    start = 2
    remain = value
    factors_list = []
    while remain != 1:
        result = remain % start
        if(result == 0):
            remain = remain / start
            factors_list.append(start)
        else:
            start += 1

    return factors_list   


def flatten(iterable):
    result = []
    for element in iterable:
        if element == None:
            continue
        if isinstance(element, int):
            result.append(element)
        else:
            result.extend(flatten(element))
    return result

def gcdOfStrings(str1, str2):
    equal = True
    max_str = ''
    min_str = ''
    if len(str1) > len(str2):
        max_str = str1
        min_str = str2
    else:
        max_str = str2
        min_str = str1

    for i,char in enumerate(min_str):
        if char != max_str[i]:
            equal = False

    if equal:
        remain = max_str[len(min_str):]
        if remain == '':
            return min_str
        else:
            return gcdOfStrings(min_str, remain)
    else:
        return ""

def kidsWithCandies(candies, extraCandies):
    result = []
    greatest = max(candies)
    for candy in candies:
        have = candy + extraCandies
        if have >= greatest:
            result.append(True)
        else:
            result.append(False)
    
    return result
    
def productExceptSelf(nums):
    result = [1] * len(nums)
    pre_num = 1
    nex_num = 1
    for i, num in enumerate(nums):
        result[i] *= pre_num
        pre_num *= num

    for i in range(len(nums) - 1, -1, -1):
        result[i] *= nex_num
        nex_num *= nums[i]
    
    return result

def increasingTriplet(nums):
    first = float('inf')
    second = float('inf')
    for num in nums:
        if num <= first:
            first = num
        elif num <= second:
            second = num
        else:
            return True
        print(num, first, second)
    return False

def compress(chars):
    chars.append("")
    cur_char = ""
    res_index = 0
    count = 0
    for num in chars:
        if not cur_char:
            cur_char = num
            count = 1
        if num != cur_char:
            chars[res_index] = cur_char
            res_index += 1

            if count > 1:
                for char_num in str(count):
                    chars[res_index] = char_num
                    res_index += 1
            cur_char = num
            count = 1
        else:
            count += 1
    
    print(chars)
    return res_index

def merge(nums1, m, nums2, n):
    for i in range(n):
        nums1[m+i] = nums2[i]
    nums1.sort()
    return nums1

def removeDuplicates(nums):
    prev_num = nums[0] + 1
    idex = 0
    while idex < len(nums): 
        if nums[idex] == prev_num:
            nums.pop(idex)
        else:
            prev_num = nums[idex]
            idex += 1

    print(nums)
    return len(nums)

def removeElement(nums, val):
    idex = 0
    while idex < len(nums): 
        if nums[idex] == val:
            nums.pop(idex)
        else:
            idex += 1 

    print(nums)
    return len(nums)

def removeDuplicates(nums):
    pre_num = nums[0]
    count = 1
    index = 1
    while index < len(nums):
        if nums[index] == pre_num:
            count += 1
        else:
            pre_num = nums[index]
            count = 1

        if count < 3:
            index += 1
        else:
            nums.pop(index)


def removeDuplicates_v2(nums):
    j = 1
    for i in range(1, len(nums)):
        if j == 1 or nums[i] != nums[j - 2]:
            nums[j] = nums[i] 
            j += 1
    print(nums)

def majorityElement(nums):
    nums.sort()
    return nums[len(nums)//2]

def rotate(nums, k):
    k = k % len(nums)
    if k != 0:
        nums[:k], nums[k:] = nums[-k:], nums[:k]
    return nums

def maxProfit(prices):
    profit = 0
    min_price = prices[0]
    max_price = prices[0]
    for i in range(len(prices)):
        if prices[i] < min_price:
            min_price = prices[i]
            max_price = prices[i]
        if prices[i] >= max_price:
            max_price = prices[i]
        
        profit = max(profit , max_price - min_price)
    return profit

def maxProfit_v2(prices):
    total_profit = 0
    profit = 0
    min_price = prices[0]
    max_price = prices[0]
    for i in range(len(prices)):
        if prices[i] < min_price:
            min_price = prices[i]
            max_price = prices[i]
        if prices[i] >= max_price:
            max_price = prices[i]
        
        if max_price - min_price > profit:
            #print(min_price, max_price)
            total_profit += max_price - min_price
            min_price = max_price

    return total_profit

def canJump(nums):
    if len(nums) == 1: return True
    goal = len(nums) - 1
    for i in range(len(nums)-1, -1, -1):
        if i + nums[i] >= goal:
            goal = i
    
    return True if goal == 0 else False

def jump(nums):
    near = far = jumps = 0
    while far < len(nums) - 1:
        farthest = 0
        for i in range(near, far + 1):
            farthest = max(farthest, i + nums[i])
        
        near = far + 1
        far = farthest
        jumps += 1
    return jumps

def hIndex(citations):
    citations.sort(reverse = True)
    h_index = 0
    for i in range(len(citations)):
        if citations[i] >= i+1:
            h_index += 1

    return h_index 

import random
class RandomizedSet(object):

    def __init__(self):
        self.index = -1
        self.set = {}
        self.reverseSet = {}
        self.list = []

    def insert(self, val):
        if val not in self.set.values():
            self.index += 1
            self.set[self.index] = val
            self.reverseSet[val] = self.index
            self.list.append(val)
            
            return True
        else: return False

    def remove(self, val):
        if val in self.set.values():
            key = self.reverseSet[val]
            self.set[key] = self.set[self.index]
            del self.reverseSet[val]
            self.reverseSet[self.set[self.index]] = key
            del self.set[self.index]

            self.list[key] = self.list[self.index]
            self.list.pop()
            self.index -= 1
            return True
        
        else: return False
    def getRandom(self):
        try:
            return random.sample(self.list, 1)[0]
        except ValueError:
            return None
import random

class RandomizedSetV2:
    def __init__(self):
        self.lst = []
        self.idx_map = {}

    def search(self, val):
        return val in self.idx_map

    def insert(self, val):
        if self.search(val):
            return False

        self.lst.append(val)
        self.idx_map[val] = len(self.lst) - 1
        return True

    def remove(self, val):
        if not self.search(val):
            return False

        idx = self.idx_map[val]
        self.lst[idx] = self.lst[-1]
        self.idx_map[self.lst[-1]] = idx
        self.lst.pop()
        del self.idx_map[val]
        return True

    def getRandom(self):
        return random.choice(self.lst)

def canCompleteCircuit(gas, cost):
    n = len(gas)
    gas.extend(gas)
    cost.extend(cost)
    start = curr = 0
    for i in range(len(gas)):
        if i == start + n:
            return start
        curr = curr + gas[i] - cost[i]
        if curr < 0:
            start = i + 1
            curr = 0

    return - 1

def convert(s, numRows):
    if numRows == 1: return s
    matrix = [[0 for _ in range(len(s))] for _ in range(numRows)]
    i = j = 0
    count = 0
    for char in s:
        if count < numRows:
            matrix[i][j] = char
            if count != numRows - 1: 
                i += 1
        else:
            if i - 1 != 0:
                i -= 1
                j += 1
                matrix[i][j] = char
            else:
                count = 0
                i -= 1
                j += 1
                matrix[i][j] = char
                i += 1

        count += 1

    return ''.join(element for row in matrix for element in row if element != 0)



#gcd(ABAB, ABABAB mod ABAB) = gcd(ABAB, AB)
#gcd(AB, ABAB mod AB) = gcd(AB, AB) 
#gcd(AB, AB mod AB) = gcd(AB, 0)

