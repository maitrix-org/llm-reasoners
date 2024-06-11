(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects d b i)
(:init 
(harmony)
(planet d)
(planet b)
(planet i)
(province d)
(province b)
(province i)
)
(:goal
(and
(craves d b)
(craves b i)
)))