import math
from collections import Counter
from dataclasses import asdict
from typing import Collection, Iterator, List, Optional

from utils.settings import settings


class BasePlanner:
    """
    Abstract class Base Planner.
    Use one of its children class.
    Global settings are changed in place, but at the end of the iteration every settings are rollback to their initial
    values.
    """

    _existing_names = Counter()

    def __init__(self, runs_name: Optional[str] = None):
        """
        Create an abstract setting planner object.

        :param runs_name: The string template to use to generate each run name. Allow specifying any setting field with
            default f-string syntaxe. If multiple run have the same name, add increment. If this parameter is None, the
            name will be generated according to the settings name, which is very descriptive but could be quite long.
            The special key "i" refer to the current run index.
            Eg:
                runs_name="run-{model_type}" => "run-CNN", "run-FF", "run-RNN", ...
                runs_name="test-run" => "test-run", "test-run-002, "test-run-003", ...
                runs_name="{i:03d}-{method}" => "01-method_a", "02-method_b", ...
        """
        self.runs_name = self.default_runs_name() if runs_name is None else runs_name
        self.is_sub_planner = False

    def __iter__(self) -> Iterator:
        """
        Create the planner iterator.
        :return: Itself
        """
        raise NotImplemented('Iteration abstract methods need to be override')

    def __next__(self) -> str:
        """
        Change the next setting in-place, according to the planner
        :return: The name of the current run
        """
        raise NotImplemented('Iteration abstract methods need to be override')

    def __len__(self) -> int:
        """
        Get the length of the planner (the total number of iterations)
        :return: The length
        """
        raise NotImplemented('Length abstract method need to be override')

    def default_runs_name(self) -> Optional[str]:
        """
        Get the default runs name if none is explicitly specify.
        :return: The default name.
        """
        raise NotImplemented('Default runs name abstract method need to be override')

    def reset_original_values(self) -> None:
        """
        Reset the setting value as it was before the start of the iteration.
        """
        raise NotImplemented('Reset abstract method need to be override')

    def reset_state(self) -> None:
        """
        Reset the planner state.
        """
        raise NotImplemented('Reset state abstract method need to be override')

    def get_formatted_name(self) -> str:
        """
        Get the final formatted run name according to the runs_name template if this is the root planner. Otherwise, is
        this is a sub planner, return the intermediate name (raw f-string).
        """
        if self.is_sub_planner:
            # Not the final name, so no formatting
            return self.runs_name

        try:
            counter_total = sum(BasePlanner._existing_names.values()) + 1
            base = self.runs_name.format_map({**asdict(settings), **{'i': counter_total}})
        except KeyError as err:
            raise KeyError(f'Invalid run name "{self.runs_name}", because the setting "{err.args[0]}" do not exist.') \
                from err

        if len(base) == 0:
            raise ValueError(f'Invalid run name "{self.runs_name}", can not be empty.')

        BasePlanner._existing_names[base] += 1
        return base if BasePlanner._existing_names[base] == 1 else f'{base}-{BasePlanner._existing_names[base]:03d}'

    def stop_iter(self):
        """
        Clean up before stop iteration, if this is the main planner (not a sub-planner that compose a bigger one)
        """
        if not self.is_sub_planner:
            self.reset_original_values()
            self.reset_state()

    @classmethod
    def reset_names(cls):
        """ Reset the existing names. """
        cls._existing_names.clear()


class Planner(BasePlanner):
    """
    Simple planner use to start a set of runs with a different values for one setting.
    Global settings are changed in place, but at the end of the iteration every settings are rollback to their initial
    values.
    """

    def __init__(self, setting_name: str, setting_values: Collection, runs_name: Optional[str] = None):
        """
        Create a simple planner that will iterate value of a specific setting.

        :param setting_name: The name of the setting to change.
        :param setting_values: The collection of value to iterate (the collection should be iterable and have a length).
        :param runs_name: The string template to use to generate each run name. Allow specifying any setting field with
            default f-string syntaxe. If multiple run have the same name, add increment. If this parameter is None, the
            name will be generated according to the settings name, which is very descriptive but could be quite long.
            The special key "i" refer to the current run index.
            Eg:
                runs_name="run-{model_type}" => "run-CNN", "run-FF", "run-RNN", ...
                runs_name="test-run" => "test-run", "test-run-002, "test-run-003", ...
                runs_name="{i:03d}-{method}" => "01-method_a", "02-method_b", ...
        """
        self.setting_name = setting_name
        self.setting_values = setting_values

        self._is_original_value = True
        self._setting_original_value = None  # Will be set when the iteration start
        self._values_iterator = None

        super().__init__(runs_name)

    def __iter__(self) -> Iterator:
        """ See :func:`~utils.planner.BasePlanner.__iter__` """
        # Save the original value if it's the first iteration
        if self._is_original_value:
            self._setting_original_value = getattr(settings, self.setting_name)
            self._is_original_value = False

        # Start values iteration
        self._values_iterator = iter(self.setting_values)
        return self

    def __next__(self) -> str:
        """ See :func:`~utils.planner.BasePlanner.__next__` """

        try:
            # Get new value
            value = next(self._values_iterator)
        except StopIteration:
            # Call last method before to trigger StopIteration
            self.stop_iter()
            raise

        # Set new value
        setattr(settings, self.setting_name, value)

        return self.get_formatted_name()

    def __len__(self) -> int:
        """ See :func:`~utils.planner.BasePlanner.__len__` """
        return len(self.setting_values)

    def default_runs_name(self) -> Optional[str]:
        """ See :func:`~utils.planner.BasePlanner.default_runs_name` """
        return f'{self.setting_name}_{{{self.setting_name}}}'

    def reset_original_values(self) -> None:
        """ See :func:`~utils.planner.BasePlanner.reset_original_values` """
        if not self._is_original_value:
            if getattr(settings, self.setting_name) != self._setting_original_value:
                setattr(settings, self.setting_name, self._setting_original_value)
            self._is_original_value = True
            self._setting_original_value = None

    def reset_state(self) -> None:
        """ See :func:`~utils.planner.BasePlanner.reset_counters` """
        self._setting_original_value = None  # Will be set when the iteration start
        self._values_iterator = None


class SequencePlanner(BasePlanner):
    """
    To organise planners by sequential order.
    When the current planner is over the next one on the list will start.
    The total length of this sequence will be the sum of each sub-planners.
    Global settings are changed in place, but at the end of the iteration every setting are rollback to their initial
    values.
    """

    def __init__(self, planners: List[BasePlanner], runs_name: Optional[str] = None):
        """
        Create a planner that will iterate some sub-planners one by one.
        The settings will be reset between each sub-planners.

        :param planners: The list of sub-planners to iterate, it can be any subclass of BasePlanner.
        :param runs_name: The string template to use to generate each run name. Allow specifying any setting field with
            default f-string syntax. If multiple run have the same name, add increment. If this parameter is None, the
            name will be generated according to the settings name, which is very descriptive but could be quite long.
            The special key "i" refer to the current run index.
            Eg:
                runs_name="run-{model_type}" => "run-CNN", "run-FF", "run-RNN", ...
                runs_name="test-run" => "test-run", "test-run-002, "test-run-003", ...
                runs_name="{i:03d}-{method}" => "01-method_a", "02-method_b", ...
        """
        if len(planners) == 0:
            raise ValueError('Empty planners list for sequence planner')

        self.planners = planners
        self._current_planner_id = 0
        self._planners_iterator = None
        self._current_planner_iterator = None

        # Tell to sub-planners who is the boss
        for p in self.planners:
            p.is_sub_planner = True

        super().__init__(runs_name)

    def __iter__(self):
        """ See :func:`~utils.planner.BasePlanner.__iter__` """
        # First iterate over every planner
        self._planners_iterator = iter(self.planners)
        self._current_planner_id = 0
        # Then iterate inside each planner
        self._current_planner_iterator = iter(next(self._planners_iterator))

        return self

    def __next__(self):
        """ See :func:`~utils.planner.BasePlanner.__next__` """
        try:
            # Try to iterate inside the current planner
            next(self._current_planner_iterator)
            return self.get_formatted_name()
        except StopIteration:
            # Reset settings as original value before next planner
            self.planners[self._current_planner_id].reset_original_values()
            self._current_planner_id += 1

            try:
                # If current planner is over, open the next one
                # If it's already the last planner then the StopIteration will be raise again here
                self._current_planner_iterator = iter(next(self._planners_iterator))
            except StopIteration:
                # Call last method before to trigger StopIteration
                self.stop_iter()
                raise

            # Recursive call with the new planner
            return next(self)

    def __len__(self):
        """ See :func:`~utils.planner.BasePlanner.__len__` """
        return sum(map(len, self.planners))

    def default_runs_name(self) -> Optional[str]:
        """ See :func:`~utils.planner.BasePlanner.default_runs_name` """
        return '-'.join(filter(None, (p.get_formatted_name() for p in self.planners)))

    def reset_original_values(self) -> None:
        """ See :func:`~utils.planner.BasePlanner.reset_original_values` """
        for p in self.planners:
            p.reset_original_values()

    def reset_state(self) -> None:
        """ See :func:`~utils.planner.BasePlanner.reset_counters` """
        self._current_planner_id = 0
        self._planners_iterator = None
        self._current_planner_iterator = None
        for p in self.planners:
            p.reset_state()


class ParallelPlanner(BasePlanner):
    """
    To organise planners in parallel.
    All planners will be applied at the same time.
    The total length of this planner will be equal to the length of the sub-planners (which should all have the same
    length).
    Global settings are changed in place, but at the end of the iteration every settings are rollback to their initial
    values.
    """

    def __init__(self, planners: List[BasePlanner], runs_name: Optional[str] = None):
        """
        Create a planner that will change the settings according to several sub-planners at the same time.
        Eg: At iteration the 5 with 2 sub-planners, the setting will be set as sub-planner[0][5] and sub-planner[1][5].
        It is not multi-thread computing.

        :param planners: The list of sub-planners to iterate in parallel, it can be any subclass of BasePlanner but they
        all need to have the same length.
        :param runs_name: The string template to use to generate each run name. Allow specifying any setting field with
            default f-string syntaxe. If multiple run have the same name, add increment. If this parameter is None, the
            name will be generated according to the settings name, which is very descriptive but could be quite long.
            The special key "i" refer to the current run index.
            Eg:
                runs_name="run-{model_type}" => "run-CNN", "run-FF", "run-RNN", ...
                runs_name="test-run" => "test-run", "test-run-002, "test-run-003", ...
                runs_name="{i:03d}-{method}" => "01-method_a", "02-method_b", ...
        """
        # Exclude AdaptativePlanner from the checking condition because its size is adaptative
        fixed_planners = [p for p in planners if not isinstance(p, AdaptativePlanner)]

        if len(fixed_planners) == 0:
            raise ValueError('Empty planners list for parallel planner')

        # Check planners length
        if not all(len(x) == len(fixed_planners[0]) for x in fixed_planners[1:]):
            raise ValueError('Impossible to run parallel planner if all sub-planners don\'t have the same length')

        # TODO forbid same setting multiple times

        self.planners: List[BasePlanner] = planners
        self._planners_iterators: List[Optional[Iterator]] = [None] * len(planners)

        # Tell to sub-planners who is the boss
        for p in self.planners:
            p.is_sub_planner = True

        super().__init__(runs_name)

    def __iter__(self):
        """ See :func:`~utils.planner.BasePlanner.__iter__` """
        # Iterate over every planner
        self._planners_iterators = [iter(p) for p in self.planners]

        return self

    def __next__(self):
        """ See :func:`~utils.planner.BasePlanner.__next__` """
        try:
            for it in self._planners_iterators:
                next(it)
        except StopIteration:
            # Call last method before to trigger StopIteration
            self.stop_iter()
            raise

        return self.get_formatted_name()

    def __len__(self):
        """ See :func:`~utils.planner.BasePlanner.__len__` """
        # Size of the first fixed planner
        return len(next(p for p in self.planners if not isinstance(p, AdaptativePlanner)))

    def default_runs_name(self) -> Optional[str]:
        """ See :func:`~utils.planner.BasePlanner.default_runs_name` """
        return '-'.join(filter(None, (p.get_formatted_name() for p in self.planners)))

    def reset_original_values(self) -> None:
        """ See :func:`~utils.planner.BasePlanner.reset_original_values` """
        for p in self.planners:
            p.reset_original_values()

    def reset_state(self) -> None:
        """ See :func:`~utils.planner.BasePlanner.reset_counters` """
        self._planners_iterators = [None] * len(self.planners)
        for p in self.planners:
            p.reset_state()


class CombinatorPlanner(BasePlanner):
    """
    To organise the combination of planners.
    Each possible combination of values from planners will be used.
    The total length will be the product of the length of each sub-planners.
    Global settings are changed in place, but at the end of the iteration every settings are rollback to their initial
    values.
    """

    def __init__(self, planners: List[BasePlanner], runs_name: Optional[str] = None):
        """
        Creat a combinator planner that will iterate over all possible combinations of sub-planners values.
        Eg: with sub-planner A iterates through [1, 2] abd sub-planner B iterates through [True, False], this planner
        will iterate through [A_1-B_True, A_2-B_True, A_1-B_False, A_2-B_False].

        :param planners: The list of sub-planners to iterate in parallel, it can be any subclass of BasePlanner but they
        all need to have the same length.
        :param runs_name: The string template to use to generate each run name. Allow specifying any setting field with
            default f-string syntaxe. If multiple run have the same name, add increment. If this parameter is None, the
            name will be generated according to the settings name, which is very descriptive but could be quite long.
            The special key "i" refer to the current run index.
            Eg:
                runs_name="run-{model_type}" => "run-CNN", "run-FF", "run-RNN", ...
                runs_name="test-run" => "test-run", "test-run-002, "test-run-003", ...
                runs_name="{i:03d}-{method}" => "01-method_a", "02-method_b", ...
        """
        if len(planners) == 0:
            raise ValueError('Empty planners list for combinator planner')

        # TODO forbid same setting multiple times

        self.planners: List[BasePlanner] = planners
        self._planners_iterators: List[Optional[Iterator]] = [None] * len(planners)
        self._first_iter = True

        # Tell to sub-planners who is the boss
        for p in self.planners:
            p.is_sub_planner = True

        super().__init__(runs_name)

    def __iter__(self):
        """ See :func:`~utils.planner.BasePlanner.__iter__` """
        # Iterate over every planner
        self._first_iter = True
        self._planners_iterators = [iter(p) for p in self.planners]

        return self

    def __next__(self):
        """ See :func:`~utils.planner.BasePlanner.__next__` """
        # For the first iteration, initialise every sub-planners with their first value
        if self._first_iter:
            self._first_iter = False
            for it in self._planners_iterators:
                next(it)

            return self.get_formatted_name()

        for i in range(len(self.planners)):
            try:
                next(self._planners_iterators[i])
                return self.get_formatted_name()
            except StopIteration:
                # If stop iteration trigger for the last sub-planner then the iteration is over, and we let error
                # propagate.
                if i == (len(self.planners) - 1):
                    # Call last method before to trigger StopIteration
                    self.stop_iter()
                    raise

                # If stop iteration trigger for an intermediate sub-planner, reset it and continue the loop
                self._planners_iterators[i] = iter(self.planners[i])
                next(self._planners_iterators[i])

    def __len__(self):
        """ See :func:`~utils.planner.BasePlanner.__len__` """
        return math.prod(map(len, self.planners))

    def default_runs_name(self) -> Optional[str]:
        """ See :func:`~utils.planner.BasePlanner.default_runs_name` """
        return '-'.join(filter(None, (p.get_formatted_name() for p in self.planners)))

    def reset_original_values(self):
        """ See :func:`~utils.planner.BasePlanner.reset_original_values` """
        for p in self.planners:
            p.reset_original_values()

    def reset_state(self) -> None:
        """ See :func:`~utils.planner.BasePlanner.reset_counters` """
        self._planners_iterators = [None] * len(self.planners)
        self._first_iter = True
        for p in self.planners:
            p.reset_state()


class AdaptativePlanner(BasePlanner):
    """
    A planner that allow to change the values of the settings depending on the other settings values.
    Should be used as a child of a ParallelPlanner.
    """

    def __init__(self, setting_name: str, setting_value_template: str, runs_name: Optional[str] = None):
        """
        Create an adaptative planner the change its values depending on the other settings values.

        :param setting_name: The name of the setting to change.
        :param setting_value_template: The string template to use to generate the setting value. Same syntaxe as for
            runs_name
        :param runs_name: The string template to use to generate each run name. Allow specifying any setting field with
            default f-string syntaxe. If multiple run have the same name, add increment. If this parameter is None, the
            name will be generated according to the settings name, which is very descriptive but could be quite long.
            The special key "i" refer to the current run index.
            Eg:
                runs_name="run-{model_type}" => "run-CNN", "run-FF", "run-RNN", ...
                runs_name="test-run" => "test-run", "test-run-002, "test-run-003", ...
                runs_name="{i:03d}-{method}" => "01-method_a", "02-method_b", ...
        """
        self.setting_name = setting_name
        self.setting_value_template = setting_value_template

        self._is_original_value = True
        self._setting_original_value = None  # Will be set when the iteration start

        super().__init__(runs_name)

    def __iter__(self) -> Iterator:
        """ See :func:`~utils.planner.BasePlanner.__iter__` """
        if not self.is_sub_planner:
            raise ValueError('Adaptative planner should only be used as a child of a ParallelPlanner.')

        # Save the original value if it's the first iteration
        if self._is_original_value:
            self._setting_original_value = getattr(settings, self.setting_name)
            self._is_original_value = False

        return self

    def __next__(self) -> str:
        """ See :func:`~utils.planner.BasePlanner.__next__` """

        counter_total = sum(BasePlanner._existing_names.values()) + 1
        # Get new value depending on the other settings values
        value = self.setting_value_template.format_map({**asdict(settings), **{'i': counter_total}})

        # Set new value
        setattr(settings, self.setting_name, value)

        return self.get_formatted_name()

    def __len__(self) -> int:
        """ See :func:`~utils.planner.BasePlanner.__len__` """
        return 0  # The size depends on the other iterators

    def default_runs_name(self) -> Optional[str]:
        """ See :func:`~utils.planner.BasePlanner.default_runs_name` """
        return f'{self.setting_name}_{{{self.setting_name}}}'

    def reset_original_values(self) -> None:
        """ See :func:`~utils.planner.BasePlanner.reset_original_values` """
        if not self._is_original_value:
            if getattr(settings, self.setting_name) != self._setting_original_value:
                setattr(settings, self.setting_name, self._setting_original_value)
            self._is_original_value = True
            self._setting_original_value = None

    def reset_state(self) -> None:
        """ See :func:`~utils.planner.BasePlanner.reset_counters` """
        self._setting_original_value = None  # Will be set when the iteration start
