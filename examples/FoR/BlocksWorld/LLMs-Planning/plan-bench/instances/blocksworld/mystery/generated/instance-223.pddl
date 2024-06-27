(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects d l i)
(:init 
(harmony)
(planet d)
(planet l)
(planet i)
(province d)
(province l)
(province i)
)
(:goal
(and
(craves d l)
(craves l i)
)))