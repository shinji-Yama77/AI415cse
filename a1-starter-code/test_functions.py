import ast
import inspect
import types
import unittest


import a1_exercises as a1
import generate as gen


class TestA1Functions(unittest.TestCase):

    def setUp(self):
        self.maxDiff = None

    def _test_function_on_cases(self, func, test_cases, test_type=False,
                                expect_type=None, assert_fn=None):
        if assert_fn is None:
            assert_fn = self.assertEqual

        for test_input, expected_output in test_cases:
            actual_output = func(test_input)
            assert_fn(
                expected_output, actual_output,
                f"{func.__name__}({test_input!r}) \n"
                f"returned: {actual_output!r}\n"
                f"expected: {expected_output!r}")
            if test_type:
                self.assertIs(
                    type(actual_output), expect_type,
                    f"Return value of {func.__name__}({test_input!r}) is "
                    f"not of type {expect_type.__name__}!")

    def test_is_triple(self):
        """Provided tests for is_triple in starter code."""
        test_cases = [
            (0, True), (144, True), (1, False), (22, False), (-1, False),
            (-15, True),
        ]
        self._test_function_on_cases(a1.is_a_triple, test_cases)

    def test_last_prime(self):
        """Provided tests for last_prime in starter code."""
        test_cases = [
            (2, 2), (3, 3), (4, 3), (10, 7), (12, 11), (100, 97),
            (200, 199),
        ]
        self._test_function_on_cases(a1.last_prime, test_cases, test_type=True,
                                     expect_type=int)

    def test_quadratic_roots(self):
        """Provided tests for quadratic_roots in starter code."""
        roots = a1.quadratic_roots(1, 4, -21)
        assert type(roots) is tuple, (
            "Value returned from quadratic_roots(1, 4, -21) is not a tuple!")
        x1, x2 = sorted(roots)
        assert type(x1) is float and type(x2) is float, (
            "Roots returned from quadratic_roots(1, 4, -21) are not floats!")
        self.assertAlmostEqual(x1, -7.0)
        self.assertAlmostEqual(x2, 3.0)

        self.assertEqual(a1.quadratic_roots(1, 1, 1), "complex")

    def test_new_quadratic_function(self):
        """Provided tests for new_quadratic_function in starter code."""
        qf1 = a1.new_quadratic_function(3, 2, 1)
        qf2 = a1.new_quadratic_function(6, 5, 4)
        result = qf1(10)
        assert(result == 321)
        result = qf2(10)
        assert(result == 654)

    def test_perfect_shuffle(self):
        """Provided tests for perfect_shuffle in starter code."""
        test_cases = [
            ([], []), ([1, 2, 3, 4], [1, 3, 2, 4]),
            ([0, 1, 2, 3, 4, 5, 6, 7], [0, 4, 1, 5, 2, 6, 3, 7]),
        ]
        self._test_function_on_cases(a1.perfect_shuffle, test_cases)

    def test_list_of_3_times_elts_plus_1(self):
        """Provided tests for list_of_3_times_elts_plus_1 in starter code."""
        test_cases = [([], []), ([1], [4]), ([1, 2, 3], [4, 7, 10])]
        self._test_function_on_cases(a1.list_of_3_times_elts_plus_1, test_cases)

    def test_list_of_3_times_elts_plus_1_use_list_comp(self):
        """Provided test for list_of_3_times_elts_plus_1 on list comprehension."""
        the_ast = ast.parse(inspect.getsource(a1.list_of_3_times_elts_plus_1))
        used_list_comp = any(
            type(node) is ast.ListComp for node in
            ast.walk(the_ast))
        self.assertTrue(
            used_list_comp, "Your did not use list comprehension when "
                            "implementing list_of_3_times_elts_plus_1!")

    def test_triple_vowels(self):
        """Provided tests for triple_vowels in starter code."""
        test_cases = [
            ("The big bad WOLF", "Theee biiig baaad WOOOLF"),
            ("The *BIG BAD* wolf!", "Theee *BIIIG BAAAD* wooolf!"),
        ]
        self._test_function_on_cases(a1.triple_vowels, test_cases)

    def test_count_words(self):
        """Provided test for count_words in starter code."""
        test_cases = [
            # Small test cases.
            (" A a a b b ", {"a": 3, "b": 2}),
            ("#screen-size: 1920*1080, 2560*1440, 1920*1080",
             {'#screen-size': 1, '1920*1080': 2, '2560*1440': 1}),
            # Natural text test cases.
            ("""Don't lie
                I want him to know
                Gods' loves die young
                Is he ready to go?
                It's the last time running through snow
                Where the vaults are full and the fire's bold
                I want to know - does it bother you?
                The low click of a ticking clock
                There's a lifetime right in front of you
                And everyone I know
                Young turks
                Young saturday night
                Young hips shouldn't break on this ice
                Old flames - they can't warm you tonight""",
             {"don't": 1, 'lie': 1, 'i': 3, 'want': 2, 'him': 1, 'to': 3,
              'know': 3, "gods'": 1, 'loves': 1, 'die': 1, 'young': 4, 'is': 1,
              'he': 1, 'ready': 1, 'go': 1, "it's": 1, 'the': 4, 'last': 1,
              'time': 1, 'running': 1, 'through': 1, 'snow': 1, 'where': 1,
              'vaults': 1, 'are': 1, 'full': 1, 'and': 2, "fire's": 1,
              'bold': 1, '-': 2, 'does': 1, 'it': 1, 'bother': 1, 'you': 3,
              'low': 1, 'click': 1, 'of': 2, 'a': 2, 'ticking': 1, 'clock': 1,
              "there's": 1, 'lifetime': 1, 'right': 1, 'in': 1, 'front': 1,
              'everyone': 1, 'turks': 1, 'saturday': 1, 'night': 1, 'hips': 1,
              "shouldn't": 1, 'break': 1, 'on': 1, 'this': 1, 'ice': 1,
              'old': 1, 'flames': 1, 'they': 1, "can't": 1, 'warm': 1,
              'tonight': 1}),
            # Synthetic test cases.
            ("2/3-1/4 A-A #x #x A-A A'B a+b a+b #x #x 91% a-a #c-d a-a a+b "
             "2/3 #c-d a-a A'B a'b a'b a-a a'b 3*2 2/3-1/4 A-A 2/3-1/4 91% #x "
             "91%",
             {'2/3-1/4': 3, 'a-a': 7, '#x': 5, "a'b": 5, 'a+b': 3, '91%': 3,
              '#c-d': 2, '2/3': 1, '3*2': 1}),
            ("cc9,\n*_\\c99]F*FF=q*'*6  9]_,=\\]F__9 q,c*9\n \n'GG9F ",
             {'cc9': 1, '*': 1, 'c99': 1, 'f*ff': 1, "q*'*6": 1, '9': 2,
              'f': 1, 'q': 1, 'c*9': 1, "'gg9f": 1}),
            ("'>\\&F\\>1W2,2 ,\n&#\\\n#W,\\\n '  1,&F& \n#\n# hF\n/\n'!/&',"
             "W \\&1,1q!W&/F!>>,qh2h #'W!W ,>&'!\n\n#q\\h\n/W#\n'!'2#\\W>\\,"
             "#q1#! 2\n1!/\\FF\\qFqF\n,,F&/1FW\\FF&\n\n'\nh'2\n/#'!\n",
             {"'": 7, 'f': 3, '1w2': 1, '2': 2, '#': 3, '#w': 1, '1': 3,
              'hf': 1, '/': 3, 'w': 4, '1q': 1, '/f': 1, 'qh2h': 1, "#'w": 1,
              '#q': 1, 'h': 1, '/w#': 1, "'2#": 1, '#q1#': 1, 'ff': 2,
              'qfqf': 1, '/1fw': 1, "h'2": 1, "/#'": 1}),
            ('f@@"\n\n\'*k00*V@ \'\'x0Vf\'\n`f _" __V\n\nf_"*\'* \'|V  *, '
             '"*xV"f k k\n"\'\'`,kx\' ,1|" x"0`kx 1\'|xx '
             '@"0Vf`1\n0"0k|ff"@0\nf0V,"1V0*_" _|"*x`V1*|f,@\'f\nk,'
             '@0\'f00f1`\'\n \nx*k,,_"k@\'00@1"x1fxx\'*k`x \n"@\',0@\'f1_V\' '
             ',k\nV0`1k|`0*@*x,1kV 1_x,@*_@111|`\n"0`_"*V*_k\'*\n_ |kx*k@` ,'
             '|1V',
             {'f@@': 1, "'*k00*v@": 1, "''x0vf'": 1, 'f': 4, 'v': 2,
              "*'*": 1,
              "'": 2, '*': 1, '*xv': 1, 'k': 4, "''": 1, "kx'": 1, '1': 3,
              'x': 3, '0': 3, 'kx': 1, "1'": 1, 'xx': 1, '@': 1, '0vf': 1,
              '0k': 1, 'ff': 1, '@0': 1, 'f0v': 1, '1v0*': 1, '*x': 1,
              'v1*': 1,
              "@'f": 1, "@0'f00f1": 1, 'x*k': 1, "k@'00@1": 1, "x1fxx'*k": 1,
              "@'": 1, "0@'f1": 1, "v'": 1, 'v0': 1, '1k': 1, '0*@*x': 1,
              '1kv': 1, '@*': 1, '@111': 1, '*v*': 1, "k'*": 1, 'kx*k@': 1,
              '1v': 1}),
            ('\n@,@:i:,-^:8.\\M3,@^ ^ +.@  8ix\n+:M-+E\\\\ : ,\n:i,\n \\^x,'
             '\\M8,\n8.+iM.,3\\3^E \n-\nM8\\3x+\\i3,'
             'E\n +@-:\\-\\\n-+^xM\n--3+\\3\\:3:@:E^^33:\nM@\\EEEE\\M+.3\\^Mx'
             '.xi+^8EE,@,iE: @x8-x.,iM.:.EiM.ME+++ i3\n \n+\n x^,:-MEx8\\M.3 '
             '^   \\3^iM:-+\n\\EM,@.Ex-,i. .,3:,3MM-+@.33i88\n,'
             '8-EEE:\\ExE8-^+8iE\\.+.\\+E@^+^@x-^8++  E .M@8,: .:^\\8\\3i.388 '
             'i3,8,:,\n\\.3.-^8^ 3E\\,xix\\.+E+E+8-\n+^i^.:-@: i8^ -@^',
             {'@': 7, 'i': 4, '-': 4, '8': 5, 'm3': 1, '+': 6, '8ix': 1,
              'm-+e': 1, 'x': 2, 'm8': 2, '+im': 1, '3': 9, 'e': 4, '3x+': 1,
              'i3': 3, '+@-': 1, '-+': 2, 'xm': 1, '--3+': 1, '33': 1,
              'm@': 1,
              'eeee': 1, 'm+': 1, 'mx': 1, 'xi+': 1, '8ee': 1, 'ie': 1,
              '@x8-x': 1, 'im': 2, 'eim': 1, 'me+++': 1, '-mex8': 1, 'm': 1,
              'em': 1, 'ex-': 1, '3mm-+@': 1, '33i88': 1, '8-eee': 1,
              'exe8-': 1, '+8ie': 1, '+e@': 1, '@x-': 1, '8++': 1, 'm@8': 1,
              '3i': 1, '388': 1, '3e': 1, 'xix': 1, '+e+e+8-': 1, '-@': 2,
              'i8': 1}),
        ]
        self._test_function_on_cases(
            a1.count_words, test_cases, test_type=True, expect_type=dict,
            assert_fn=self.assertDictEqual)

    def test_generate(self):
        gen.CHOICE_MODE = 'first'

        text = gen.gen_overall_message()
        self.assertEqual("went away lol.  went away lol.  ", text)


if __name__ == '__main__':
    unittest.main()
