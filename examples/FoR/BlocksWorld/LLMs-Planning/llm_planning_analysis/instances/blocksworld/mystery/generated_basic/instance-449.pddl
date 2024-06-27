

(define (problem MY-rand-5)
(:domain mystery-4ops)
(:objects a b c d e )
(:init
(harmony)
(craves a c)
(planet b)
(craves c d)
(planet d)
(planet e)
(province a)
(province b)
(province e)
)
(:goal
(and
(craves b d)
(craves c e)
(craves e a))
)
)


