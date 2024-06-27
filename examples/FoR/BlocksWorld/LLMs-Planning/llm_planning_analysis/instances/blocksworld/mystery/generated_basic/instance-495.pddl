

(define (problem MY-rand-5)
(:domain mystery-4ops)
(:objects a b c d e )
(:init
(harmony)
(planet a)
(craves b d)
(planet c)
(craves d e)
(craves e c)
(province a)
(province b)
)
(:goal
(and
(craves a b)
(craves b d))
)
)


