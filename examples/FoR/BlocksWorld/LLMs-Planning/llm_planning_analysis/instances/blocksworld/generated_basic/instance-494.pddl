

(define (problem BW-rand-5)
(:domain blocksworld-4ops)
(:objects a b c d e )
(:init
(handempty)
(ontable a)
(ontable b)
(on c b)
(ontable d)
(ontable e)
(clear a)
(clear c)
(clear d)
(clear e)
)
(:goal
(and
(on d c)
(on e a))
)
)


