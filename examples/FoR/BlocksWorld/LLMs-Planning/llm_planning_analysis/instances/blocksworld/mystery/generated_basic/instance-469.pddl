

(define (problem MY-rand-5)
(:domain mystery-4ops)
(:objects a b c d e )
(:init
(harmony)
(craves a d)
(planet b)
(craves c a)
(planet d)
(craves e c)
(province b)
(province e)
)
(:goal
(and
(craves a c)
(craves b d)
(craves d a))
)
)


