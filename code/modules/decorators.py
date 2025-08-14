from functools import wraps
from modules.email import Email
from typing import Callable, Any

def output_state_warning(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Decorator warning respect to the success or failure of the 
    function passed.

    Args:
        - func (Callable): Function to be passed 
    """
    def construct_message(*args, exception: bool, e: None | Exception = None, **kwargs: Any) -> tuple[str, str]:
        """
        Constructs the message based on the case: Exception raised or not.

        Args:
            - exception (bool): True if exception is being raised, False otherwise.
            - e (NoneType | Exception): Exception raised if any.

        Returns:
            - tuple[str, str]: Tuple of strings, where the first one is the subject to the email and the second one is the message.
        """
        if exception == True:
            subject = "EXCEPTION while running process!"
            main_message = f"There has been an exception with the following details: \n\t- Type: {type(e)} \n\t- Message: {e} \n\t- Args: {e.args} \n\n Details (arguments) of the failed process {func.__name__} are:"
        else:
            subject = "EXECUTION finished!"
            main_message = "Processed successfully completed. Details (arguments) are:"
        pieces = [main_message]
        pieces.extend(str(arg) for arg in args)
        pieces.extend(f"{k}={v}" for k, v in kwargs.items())
        message = "\n\t- ".join(pieces)
        print(message)
        return subject, message

    @wraps(func)
    def wrapper(*args, **kwargs):
        """
        Returned extended function by the decorator.

        Args:
            - 
        """
        email = Email()
        try:
            result = func(*args, **kwargs)
        except Exception as e:
            subject, message = construct_message(exception=True, e=e, *args, **kwargs)
            email.send_mail(subject, message)
            raise
        else:
            subject, message = construct_message(exception=False, *args, **kwargs)
            email.send_mail(subject, message)
            return result
    return wrapper