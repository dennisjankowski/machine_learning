"""

+++ SKIPPING FIRST ENTRY BY INITIALIZING A "LOOSE" ITERATOR +++
Initialize an iterator, skip the first entry and add it to the new list.

"""
integer_list = [1, 2, 3, 4, 5]
# initialize a loose iterator of the list 'integer_list'
iter_list = iter(integer_list)
# skip the first entry
next(iter_list)

list_without_first_item = []
for entry in iter_list:
    list_without_first_item.append(entry)
print(list_without_first_item)
