

(define (problem BW-rand-5)
(:domain blocksworld-4ops)
(:objects a b c d e )
(:init
(handempty)
(on a c)
(ontable b)
(on c d)
(ontable d)
(ontable e)
(clear a)
(clear b)
(clear e)
)
(:goal
(and
(on b d)
(on c e)
(on e a))
)
)


