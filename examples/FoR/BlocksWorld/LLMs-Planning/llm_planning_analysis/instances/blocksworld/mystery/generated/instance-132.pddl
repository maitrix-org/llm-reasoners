(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects b d e g)
(:init 
(harmony)
(planet b)
(planet d)
(planet e)
(planet g)
(province b)
(province d)
(province e)
(province g)
)
(:goal
(and
(craves b d)
(craves d e)
(craves e g)
)))