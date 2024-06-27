(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects d l j)
(:init 
(harmony)
(planet d)
(planet l)
(planet j)
(province d)
(province l)
(province j)
)
(:goal
(and
(craves d l)
(craves l j)
)))