import re

class LaTeXTokenizer:
    def __init__(self):
        self.common_tokens = [
            ' ',
            '\\begin{equation}', '\\end{equation}',
            '\\begin{align}', '\\end{align}',
            '\\begin{aligned}', '\\end{aligned}',
            '\\begin{array}', '\\end{array}',
            '\\begin{split}', '\\end{split}',
            '\\begin{gather}', '\\end{gather}',
            '\\begin{gathered}', '\\end{gathered}',
            '\\begin{cases}', '\\end{cases}',
            '\\begin{matrix}', '\\end{matrix}',
            '\\begin{pmatrix}', '\\end{pmatrix}',
            '\\begin{bmatrix}', '\\end{bmatrix}',
            '\\begin{vmatrix}', '\\end{vmatrix}',
            '\\frac', '\\sqrt', '\\sum', '\\prod', '\\int', '\\lim',
            '\\sin', '\\cos', '\\tan', '\\log', '\\ln', '\\exp',
            '\\arcsin', '\\arccos', '\\arctan',
            '\\sinh', '\\cosh', '\\tanh',
            '\\min', '\\max', '\\sup', '\\inf',
            '\\det', '\\dim', '\\ker', '\\deg',
            '\\alpha', '\\beta', '\\gamma', '\\delta', '\\epsilon', '\\zeta',
            '\\eta', '\\theta', '\\iota', '\\kappa', '\\lambda', '\\mu',
            '\\nu', '\\xi', '\\pi', '\\rho', '\\sigma', '\\tau',
            '\\upsilon', '\\phi', '\\chi', '\\psi', '\\omega',
            '\\Gamma', '\\Delta', '\\Theta', '\\Lambda', '\\Xi',
            '\\Pi', '\\Sigma', '\\Phi', '\\Psi', '\\Omega',
            '\\cdot', '\\times', '\\div', '\\pm', '\\mp',
            '\\leq', '\\geq', '\\neq', '\\approx', '\\equiv',
            '\\sim', '\\propto', '\\infty', '\\partial', '\\nabla',
            '\\exists', '\\forall', '\\in', '\\notin',
            '\\subset', '\\subseteq', '\\cap', '\\cup', '\\emptyset',
            '\\dots', '\\ldots', '\\cdots', '\\vdots', '\\ddots',
            '\\left', '\\right',
            '\\bigl', '\\bigr', '\\Bigl', '\\Bigr',
            '\\biggl', '\\biggr', '\\Biggl', '\\Biggr',
            '\\\\', '&', '\\quad', '\\qquad',
            '\\,', '\\:', '\\;', '\\!',
            '\\text', '\\mathrm', '\\mathbf', '\\mathit'
        ]
        
        common_pattern = '|'.join(re.escape(token) for token in sorted(self.common_tokens, key=len, reverse=True))
        
        self.pattern = re.compile(
            rf'{common_pattern}|'
            r'\\[a-zA-Z]+|'
            r'\\.|'
            r'[a-zA-Z]|'
            r'[0-9]|'
            r'\S'
        )

    def tokenize(self, latex_string: str) -> list:
        tokens = self.pattern.findall(latex_string)
        return tokens