import json
import os
from pystrand.loggers.base import BaseLogger

class RunDetails(BaseLogger):
# pylint: disable=protected-access
    def save_run_details(self, optimizer):
        def _get_mutation_details(mutation_op):
            details = {
                'op_name': mutation_op.__class__.__name__
            }
            if mutation_op._mutation_probability:
                details['mutation_probability'] = mutation_op._mutation_probability

            return details

        def _get_selection_details(selection_op):
            details = {
                'op_name': selection_op.__class__.__name__
            }
            if '_selected_population_fraction' in dir(selection_op):
                details['population_fraction'] = selection_op._selected_population_fraction

            if '_selection_prob' in dir(selection_op):
                details['selection_prob'] = selection_op._selection_prob

            return details

        run_details = {
            'id': optimizer.optimizer_uuid,
            'max_iterations': optimizer._max_iterations,
            'mutation_ops': [_get_mutation_details(op) for op in optimizer._mutation_ops],
            'selection_ops': [_get_selection_details(op) for op in optimizer._selection_methods],
            'best_individual': "{}".format(optimizer.population.retrieve_best()[0])
        }
        details_path = optimizer.optimizer_uuid + ".json"
        details_path = os.path.join(self.log_path, details_path)

        with open(details_path, 'w') as file:
            json.dump(run_details, file)
