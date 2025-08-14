# Created 2025-03-25

"""Providing function wrappers for physics modifiers."""
import inspect
import functools
from collections.abc import Callable


def update(update_strategy: Callable | None = None, **kwargs):
    """Decorator to update the physics modifier function.

    Args:
        update_strategy: The update strategy to use. If None, the function will be updated by default ()
        **kwargs: Additional arguments to pass to the update strategy.

    Returns:
        A decorator that updates the physics modifier function.
    """
    def decorator(func: Callable):
        sig = inspect.signature(func)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            bound_args = sig.bind_partial(*args, **kwargs)

            # Apply overrides first to arguments not explicitly provided
            for param_key, override_val in wrapper._overrides.items():
                if param_key not in bound_args.arguments:
                    bound_args.arguments[param_key] = override_val

            bound_args.apply_defaults()
            return func(*bound_args.args, **bound_args.kwargs)

        wrapper._overrides = {}

        def update_func(**new_kwargs):
            if update_strategy:
                wrapper._overrides = update_strategy(wrapper._overrides, new_kwargs)
            else:
                wrapper._overrides.update(new_kwargs)

        wrapper.update = update_func
        wrapper._is_update_decorated = True

        return wrapper
    return decorator


if __name__ == "__main__":
    def scaling_modifier(current_overrides: dict, new_kwargs: dict) -> dict:
        updated = dict(current_overrides)
        if not new_kwargs and updated:  # No arguments provided, scale existing overrides
            for k in updated:
                updated[k] *= 0.6
        elif new_kwargs:  # New arguments provided, scale them directly
            for k, v in new_kwargs.items():
                if isinstance(v, (int, float)):
                    updated[k] = v * 0.6
                else:
                    raise ValueError(f"Scaling supports numeric only, got {k}={v}")
        return updated


    @update(update_strategy=scaling_modifier)
    def test_func(a, b=2, c=3):
        print(f"a: {a}, b: {b}, c: {c}")

    @update()
    def test_func_2(a, b=2, c=3):
        print(f"a: {a}, b: {b}, c: {c}")

    test_func(1)
    test_func_2(1)
    test_func.update(b=4)
    test_func_2.update(b=4)
    test_func(1)
    test_func_2(1)
    test_func.update(c=5)
    test_func_2.update(c=5)
    test_func(1)
    test_func_2(1)
    test_func.update(a=6, b=7)
    test_func_2.update(a=6, b=7)
    test_func()
    test_func_2()