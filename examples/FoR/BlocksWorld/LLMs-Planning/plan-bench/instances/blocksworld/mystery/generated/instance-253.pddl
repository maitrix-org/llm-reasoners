(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects b d a)
(:init 
(harmony)
(planet b)
(planet d)
(planet a)
(province b)
(province d)
(province a)
)
(:goal
(and
(craves b d)
(craves d a)
)))