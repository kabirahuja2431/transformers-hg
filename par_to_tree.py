# Python3 program to conStruct a
# binary tree from the given String

# Helper class that allocates a new node


class newNode:
    def __init__(self, data):
        self.data = data
        self.left = self.right = None


# This function is here just to test


def preOrder(node):
    if node == None:
        return
    print(node.data, end=" ")
    preOrder(node.left)
    preOrder(node.right)


# function to return the index of
# close parenthesis


def findIndex(Str, si, ei):
    if si > ei:
        return -1

    # Inbuilt stack
    s = []
    for i in range(si, ei + 1):
        # if open parenthesis, push it
        if Str[i] == "(":
            s.append(Str[i])

        # if close parenthesis
        elif Str[i] == ")":
            if s[-1] == "(":
                s.pop(-1)

                # if stack is empty, this is
                # the required index
                if len(s) == 0:
                    return i
    # if not found return -1
    return -1


# function to conStruct tree from String


def treeFromString(Str, si, ei):
    # Base case
    if si > ei:
        return None

    # new root
    root = newNode(ord(Str[si]) - ord("0"))
    index = -1

    # if next char is '(' find the
    # index of its complement ')'
    if si + 1 <= ei and Str[si + 1] == "(":
        index = findIndex(Str, si + 1, ei)

    # if index found
    if index != -1:
        # call for left subtree
        root.left = treeFromString(Str, si + 2, index - 1)

        # call for right subtree
        root.right = treeFromString(Str, index + 2, ei - 1)
    return root


# Driver Code
if __name__ == "__main__":
    Str = "(1 (8 /doesn't) (1 (3 (12 her/her) (2 (6 vulture/vulture) (0 (9 doesn't/) (0 (7 entertain/entertain) (3 (12 your/your) (2 (6 xylophone/xylophone) (0 (5 tht/that) (3 (4 does/does) (11 giggle/giggle))))))))) (13 ./?))))"
    # print(Str)
    root = treeFromString(Str, 0, len(Str) - 1)
    (preOrder(root))

# This code is contributed by pranchalK
