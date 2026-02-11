"""
Text Correction Module
Correct spelling and grammar in recognized text
"""

try:
    from spellchecker import SpellChecker
    SPELLCHECKER_AVAILABLE = True
except ImportError:
    SPELLCHECKER_AVAILABLE = False
    print("⚠ Warning: pyspellchecker not installed. Spell checking disabled.")

try:
    import language_tool_python
    GRAMMAR_CHECKER_AVAILABLE = True
except ImportError:
    GRAMMAR_CHECKER_AVAILABLE = False
    print("⚠ Warning: language-tool-python not installed. Grammar checking disabled.")


class TextCorrector:
    """Correct text spelling and grammar"""

    def __init__(self, enable_spell_check=True, enable_grammar_check=False):
        """
        Initialize text corrector

        Args:
            enable_spell_check: Enable spell checking
            enable_grammar_check: Enable grammar checking (slower)
        """
        self.spell_checker = None
        self.grammar_checker = None

        if enable_spell_check and SPELLCHECKER_AVAILABLE:
            try:
                self.spell_checker = SpellChecker()
                print("✓ Spell checker initialized")
            except Exception as e:
                print(f"⚠ Could not initialize spell checker: {e}")

        if enable_grammar_check and GRAMMAR_CHECKER_AVAILABLE:
            try:
                self.grammar_checker = language_tool_python.LanguageTool(
                    'en-US')
                print("✓ Grammar checker initialized")
            except Exception as e:
                print(f"⚠ Could not initialize grammar checker: {e}")

    def correct_spelling(self, text):
        """
        Correct spelling errors

        Args:
            text: Input text

        Returns:
            corrected_text: Text with spelling corrections
        """
        if not self.spell_checker:
            return text

        words = text.split()
        corrected_words = []

        for word in words:
            # Skip if not alphabetic
            if not word.isalpha():
                corrected_words.append(word)
                continue

            # Get correction
            corrected = self.spell_checker.correction(word)
            if corrected:
                corrected_words.append(corrected)
            else:
                corrected_words.append(word)

        return ' '.join(corrected_words)

    def correct_grammar(self, text):
        """
        Correct grammar errors

        Args:
            text: Input text

        Returns:
            corrected_text: Text with grammar corrections
        """
        if not self.grammar_checker:
            return text

        try:
            matches = self.grammar_checker.check(text)
            corrected_text = language_tool_python.utils.correct(text, matches)
            return corrected_text
        except Exception as e:
            print(f"⚠ Grammar correction error: {e}")
            return text

    def correct_text(self, text):
        """
        Apply all corrections

        Args:
            text: Input text

        Returns:
            corrected_text: Fully corrected text
        """
        if not text or text.strip() == "":
            return text

        # Spell correction
        text = self.correct_spelling(text)

        # Grammar correction
        text = self.correct_grammar(text)

        return text

    def __del__(self):
        """Cleanup"""
        if self.grammar_checker:
            try:
                self.grammar_checker.close()
            except:
                pass
