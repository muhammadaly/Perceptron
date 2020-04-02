class ListItem:
    def __init__(self, p_id):
        self.id = p_id


class List:
    def __init__(self):
        self.data = []

    def add_item(self, list_item):
        self.data.append(list_item)

    def get_by_index(self, p_index):
        return self.data[p_index]

    def get_by_id(self, p_id):
        for item in self.data:
            if item.id == p_id:
                return item
