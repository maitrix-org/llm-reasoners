(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects h b f)
(:init 
(harmony)
(planet h)
(planet b)
(planet f)
(province h)
(province b)
(province f)
)
(:goal
(and
(craves h b)
(craves b f)
)))