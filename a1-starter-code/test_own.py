import unittest
import a1_exercises as a1


class Testa1(unittest.TestCase):

    def test_count_words(self):
        t = "hello]world "
        print(a1.count_words(t))
        #dict1 = {'cc9': 1, '*': 1, 'c99': 1, 'f*ff': 1, "q*'*6": 1, '9': 2,
              #'f': 1, 'q': 1, 'c*9': 1, "'gg9f": 1}
        #dict2 = a1.count_words(t)
        #self.assertDictEqual(dict1, dict2)
        
if __name__ == '__main__':
    unittest.main()
