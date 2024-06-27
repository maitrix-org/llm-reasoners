(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects d b f)
(:init 
(harmony)
(planet d)
(planet b)
(planet f)
(province d)
(province b)
(province f)
)
(:goal
(and
(craves d b)
(craves b f)
)))